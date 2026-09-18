from __future__ import annotations

from dataclasses import dataclass
from hawk_derm.config import load_yaml


@dataclass(frozen=True)
class EncoderSpec:
    key: str
    display_name: str
    backend: str
    model_id: str
    revision: str
    dimension: int
    default_batch_size: int
    model_path: str | None = None
    import_bank: str | None = None


def load_encoder_registry(path: str = "configs/encoders.yaml") -> dict[str, EncoderSpec]:
    source = load_yaml(path)["encoders"]
    return {key: EncoderSpec(key=key, **value) for key, value in source.items()}
