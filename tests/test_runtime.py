from pathlib import Path

from hawk_derm.config import load_study_config, path_from
from hawk_derm.runtime import cpu_environment


def test_cpu_environment_sets_one_explicit_budget() -> None:
    environment = cpu_environment(1, {"EXISTING": "value"})
    assert environment["EXISTING"] == "value"
    assert environment["HAWK_DERM_CPU_WORKERS"] == "1"
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["MKL_NUM_THREADS"] == "1"
    assert environment["OPENBLAS_NUM_THREADS"] == "1"


def test_raw_data_root_environment_override(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "scin"
    monkeypatch.setenv("HAWK_DERM_RAW_DATA_ROOT", str(root))
    config = load_study_config("configs/study_25class.yaml")
    assert path_from(config, "raw_dataset_root") == root
    assert path_from(config, "raw_images_dir") == root / "images"
    assert path_from(config, "raw_metadata_csv") == root / "metadata.csv"
