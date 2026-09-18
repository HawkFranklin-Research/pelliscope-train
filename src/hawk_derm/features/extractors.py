from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile

from hawk_derm.features.bank import FeatureBank, save_feature_bank
from hawk_derm.features.registry import EncoderSpec
from hawk_derm.io import sha256_file, write_csv

ImageFile.LOAD_TRUNCATED_IMAGES = True


def resolve_device(choice: str) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def shard_name(image_id: str, image_path: str) -> str:
    return hashlib.sha256(f"{image_id}\0{image_path}".encode()).hexdigest()[:32] + ".npz"


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _torchvision_components(spec: EncoderSpec, device: torch.device) -> tuple[Callable[[list[Image.Image]], np.ndarray], str]:
    import torchvision.models as models

    if spec.key == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = torch.nn.Identity()
    elif spec.key == "inception_v3":
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        model = models.inception_v3(weights=weights, aux_logits=True)
        model.fc = torch.nn.Identity()
    else:
        raise ValueError(f"Unsupported torchvision encoder: {spec.key}")
    transform = weights.transforms()
    model.to(device).eval()

    def encode(images: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([transform(image) for image in images]).to(device)
        with torch.inference_mode():
            output = model(batch)
            if isinstance(output, tuple):
                output = output[0]
        return output.detach().float().cpu().numpy().astype(np.float32)

    return encode, str(weights)


def _transformers_components(spec: EncoderSpec, device: torch.device) -> tuple[Callable[[list[Image.Image]], np.ndarray], str]:
    from transformers import AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection

    processor = AutoImageProcessor.from_pretrained(spec.model_id, revision=spec.revision)
    model_class = CLIPVisionModelWithProjection if spec.key == "clip_vitb32" else AutoModel
    model = model_class.from_pretrained(spec.model_id, revision=spec.revision)
    if spec.key == "siglip2_so400m" and hasattr(model, "vision_model"):
        model = model.vision_model
    model = model.to(device).eval()

    def encode(images: list[Image.Image]) -> np.ndarray:
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs)
            if getattr(output, "image_embeds", None) is not None:
                features = output.image_embeds
            elif getattr(output, "pooler_output", None) is not None:
                features = output.pooler_output
            else:
                features = output.last_hidden_state[:, 0]
        return features.detach().float().cpu().numpy().astype(np.float32)

    return encode, f"{spec.model_id}@{spec.revision}"


def _derm_foundation_components(spec: EncoderSpec) -> tuple[Callable[[list[Image.Image]], np.ndarray], str]:
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError("Install the derm-foundation optional dependencies to use this encoder") from error
    model_path = Path(spec.model_path or "")
    if not model_path.exists():
        raise FileNotFoundError(f"Derm Foundation SavedModel not found: {model_path}")
    model = tf.saved_model.load(str(model_path))
    signature = model.signatures["serving_default"]
    input_name = next(iter(signature.structured_input_signature[1]))

    def serialize(image: Image.Image) -> bytes:
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        example = tf.train.Example(
            features=tf.train.Features(feature={"image/encoded": tf.train.Feature(bytes_list=tf.train.BytesList(value=[buffer.getvalue()]))})
        )
        return example.SerializeToString()

    def encode(images: list[Image.Image]) -> np.ndarray:
        batch = tf.constant([serialize(image) for image in images])
        output = signature(**{input_name: batch})
        tensor = output.get("embedding", next(iter(output.values())))
        return np.asarray(tensor, dtype=np.float32)

    return encode, str(model_path)


def extract_feature_bank(
    spec: EncoderSpec,
    image_manifest: pd.DataFrame,
    output_path: str | Path,
    *,
    device: str = "auto",
    batch_size: int | None = None,
    force: bool = False,
) -> FeatureBank:
    output_path = Path(output_path)
    shard_dir = output_path.parent / f"{spec.key}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    torch_device = resolve_device(device)
    if spec.backend == "torchvision":
        encode, resolved_revision = _torchvision_components(spec, torch_device)
    elif spec.backend == "transformers":
        encode, resolved_revision = _transformers_components(spec, torch_device)
    elif spec.backend == "tensorflow_savedmodel":
        encode, resolved_revision = _derm_foundation_components(spec)
    else:
        raise ValueError(f"Unsupported encoder backend: {spec.backend}")

    frame = image_manifest[image_manifest["retained_for_bag"].astype(bool)].copy()
    frame = frame.sort_values(["case_id", "bag_slot", "image_id"], kind="stable").reset_index(drop=True)
    failures: list[dict[str, Any]] = []
    pending: list[tuple[pd.Series, Path]] = []
    for _, row in frame.iterrows():
        shard = shard_dir / shard_name(str(row["image_id"]), str(row["resolved_image_path"]))
        if force or not shard.is_file():
            pending.append((row, shard))

    size = int(batch_size or spec.default_batch_size)
    for start in range(0, len(pending), size):
        loaded: list[tuple[pd.Series, Path, Image.Image]] = []
        for row, shard in pending[start : start + size]:
            try:
                loaded.append((row, shard, load_rgb(row["resolved_image_path"])))
            except Exception as error:
                failures.append({"case_id": row["case_id"], "image_id": row["image_id"], "error": repr(error)})
        if not loaded:
            continue
        try:
            features = encode([item[2] for item in loaded])
            if features.ndim != 2 or features.shape[1] != spec.dimension:
                raise ValueError(f"Expected (*, {spec.dimension}) embeddings, received {features.shape}")
            for (row, shard, _), feature in zip(loaded, features, strict=True):
                np.savez_compressed(shard, embedding=feature.astype(np.float32))
        except Exception as error:
            for row, _, _ in loaded:
                failures.append({"case_id": row["case_id"], "image_id": row["image_id"], "error": repr(error)})

    embeddings: list[np.ndarray] = []
    metadata: list[pd.Series] = []
    for _, row in frame.iterrows():
        shard = shard_dir / shard_name(str(row["image_id"]), str(row["resolved_image_path"]))
        if not shard.is_file():
            continue
        embeddings.append(np.load(shard, allow_pickle=False)["embedding"].astype(np.float32))
        metadata.append(row)
    if not embeddings:
        raise RuntimeError(f"No features were produced for {spec.key}")
    hashes = [str(row.get("sha256", "")) or sha256_file(row["resolved_image_path"]) for row in metadata]
    bank = FeatureBank(
        embeddings=np.stack(embeddings),
        case_ids=np.asarray([str(row["case_id"]) for row in metadata]),
        image_ids=np.asarray([str(row["image_id"]) for row in metadata]),
        image_paths=np.asarray([str(row["resolved_image_path"]) for row in metadata]),
        image_sha256=np.asarray(hashes),
        encoder=spec.key,
        model_id=spec.model_id,
        revision=resolved_revision,
        dimension=spec.dimension,
    )
    save_feature_bank(
        output_path,
        bank,
        {
            "encoder": spec.key,
            "model_id": spec.model_id,
            "requested_revision": spec.revision,
            "resolved_revision": resolved_revision,
            "dimension": spec.dimension,
            "dtype": "float32",
            "device": str(torch_device) if spec.backend != "tensorflow_savedmodel" else "tensorflow-default",
            "failure_count": len(failures),
        },
    )
    write_csv(output_path.parent / f"{spec.key}_extraction_failures.csv", pd.DataFrame(failures))
    return bank
