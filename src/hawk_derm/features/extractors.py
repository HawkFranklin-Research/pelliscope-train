from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
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


def _torchvision_components(
    spec: EncoderSpec, device: torch.device, workers: int
) -> tuple[Callable[[list[Image.Image]], np.ndarray], str]:
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
        with ThreadPoolExecutor(max_workers=min(workers, len(images))) as pool:
            transformed = list(pool.map(transform, images))
        batch = torch.stack(transformed).to(device)
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
        # Hugging Face BiT returns pooled features as (N, C, 1, 1), while
        # ViT/CLIP/SigLIP return (N, C). Normalize singleton spatial outputs to
        # the common two-dimensional feature-bank schema.
        if features.ndim > 2 and int(np.prod(features.shape[1:])) == spec.dimension:
            features = features.reshape(features.shape[0], spec.dimension)
        return features.detach().float().cpu().numpy().astype(np.float32)

    return encode, f"{spec.model_id}@{spec.revision}"


def _derm_foundation_components(
    spec: EncoderSpec, workers: int
) -> tuple[Callable[[list[Image.Image]], np.ndarray], str]:
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError("Install the derm-foundation optional dependencies to use this encoder") from error
    # The official Derm Foundation SavedModel contains a StableHLO module
    # compiled for CPU. If TensorFlow sees a CUDA device it otherwise places
    # the signature on GPU and fails with "platform CUDA ... required [CPU]".
    # Hide TensorFlow GPUs before loading the model; PyTorch encoder jobs run in
    # separate processes and remain free to use CUDA.
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as error:
        raise RuntimeError("Derm Foundation requires TensorFlow CPU placement before GPU initialization") from error
    tf.config.threading.set_intra_op_parallelism_threads(workers)
    tf.config.threading.set_inter_op_parallelism_threads(min(4, workers))
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
        with ThreadPoolExecutor(max_workers=min(workers, len(images))) as pool:
            serialized = list(pool.map(serialize, images))
        batch = tf.constant(serialized)
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
    workers: int = 1,
    force: bool = False,
) -> FeatureBank:
    output_path = Path(output_path)
    shard_dir = output_path.parent / f"{spec.key}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    torch_device = resolve_device(device)
    if spec.backend == "torchvision":
        encode, resolved_revision = _torchvision_components(spec, torch_device, workers)
    elif spec.backend == "transformers":
        encode, resolved_revision = _transformers_components(spec, torch_device)
    elif spec.backend == "tensorflow_savedmodel":
        encode, resolved_revision = _derm_foundation_components(spec, workers)
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
    def load_pending(item: tuple[pd.Series, Path]) -> tuple[pd.Series, Path, Image.Image | None, str | None]:
        row, shard = item
        try:
            return row, shard, load_rgb(row["resolved_image_path"]), None
        except Exception as error:  # image failures belong in the extraction ledger
            return row, shard, None, repr(error)

    def save_shard(item: tuple[pd.Series, Path, Image.Image, np.ndarray]) -> None:
        _, shard, _, feature = item
        np.savez_compressed(shard, embedding=feature.astype(np.float32))

    def load_shard(item: tuple[pd.Series, Path]) -> tuple[pd.Series, np.ndarray] | None:
        row, shard = item
        if not shard.is_file():
            return None
        return row, np.load(shard, allow_pickle=False)["embedding"].astype(np.float32)

    decode_pool = ThreadPoolExecutor(max_workers=max(1, workers))
    for start in range(0, len(pending), size):
        loaded: list[tuple[pd.Series, Path, Image.Image]] = []
        for row, shard, image, error in decode_pool.map(load_pending, pending[start : start + size]):
            if error is not None or image is None:
                failures.append({"case_id": row["case_id"], "image_id": row["image_id"], "error": error})
            else:
                loaded.append((row, shard, image))
        if not loaded:
            continue
        try:
            features = encode([item[2] for item in loaded])
            if features.ndim != 2 or features.shape[1] != spec.dimension:
                raise ValueError(f"Expected (*, {spec.dimension}) embeddings, received {features.shape}")
            list(decode_pool.map(save_shard, [(*item, feature) for item, feature in zip(loaded, features, strict=True)]))
        except Exception as error:
            for row, _, _ in loaded:
                failures.append({"case_id": row["case_id"], "image_id": row["image_id"], "error": repr(error)})
    embeddings: list[np.ndarray] = []
    metadata: list[pd.Series] = []
    shard_items = [
        (row, shard_dir / shard_name(str(row["image_id"]), str(row["resolved_image_path"])))
        for _, row in frame.iterrows()
    ]
    for loaded_shard in decode_pool.map(load_shard, shard_items):
        if loaded_shard is not None:
            row, embedding = loaded_shard
            metadata.append(row)
            embeddings.append(embedding)
    decode_pool.shutdown()
    write_csv(output_path.parent / f"{spec.key}_extraction_failures.csv", pd.DataFrame(failures))
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
            "device": str(torch_device) if spec.backend != "tensorflow_savedmodel" else "tensorflow-cpu",
            "workers": workers,
            "failure_count": len(failures),
        },
    )
    return bank
