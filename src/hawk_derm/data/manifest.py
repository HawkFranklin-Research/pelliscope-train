from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from hawk_derm.config import path_from
from hawk_derm.constants import CASE_ID_CANDIDATES, PATH_CANDIDATES, slugify
from hawk_derm.io import write_csv, write_json


def _first_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lowered = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _parse_sequence(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return [piece.strip() for piece in text.split("|") if piece.strip()]


def _label_column(frame: pd.DataFrame, label: str) -> str | None:
    slug = slugify(label)
    aliases = {
        slug,
        f"is_{slug}",
        f"label_{slug}",
        label,
        f"is_{label}",
    }
    normalized = {slugify(str(column)): str(column) for column in frame.columns}
    for alias in aliases:
        candidate = normalized.get(slugify(alias))
        if candidate is not None:
            return candidate
    return None


def resolve_image_path(raw_value: Any, raw_root: Path, raw_images_dir: Path) -> Path:
    value = str(raw_value).strip()
    candidate = Path(value).expanduser()
    basename = candidate.name
    relative_variants = [
        candidate,
        Path(value.replace("\\", "/")),
        Path(*Path(value.replace("\\", "/")).parts[-3:]),
        Path(*Path(value.replace("\\", "/")).parts[-2:]),
        Path(basename),
    ]
    roots = [Path("/"), raw_root, raw_root / "images", raw_images_dir, raw_images_dir / "dataset" / "images"]
    seen: set[Path] = set()
    for root in roots:
        for relative in relative_variants:
            path = relative if relative.is_absolute() else root / relative
            try:
                path = path.resolve(strict=False)
            except OSError:
                continue
            if path in seen:
                continue
            seen.add(path)
            if path.is_file():
                return path
    return (raw_images_dir / basename).resolve(strict=False)


def build_manifests(config: dict[str, Any], run_mode: str = "full") -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = list(config["labels"])
    parquet_path = path_from(config, "legacy_25class_parquet")
    legacy_cases_path = path_from(config, "legacy_case_manifest")
    legacy_images_path = path_from(config, "legacy_image_manifest")
    required = [parquet_path, legacy_cases_path, legacy_images_path]
    missing_sources = [str(path) for path in required if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(
            "The September 25-class cohort sources are required before manifest construction. "
            f"Missing: {missing_sources}. Configure HAWK_DERM_PATH_LEGACY_25CLASS_PARQUET, "
            "HAWK_DERM_PATH_LEGACY_CASE_MANIFEST, and HAWK_DERM_PATH_LEGACY_IMAGE_MANIFEST on a deployment VM."
        )
    if run_mode == "full":
        raw_images_dir = path_from(config, "raw_images_dir")
        available_pngs = (
            sum(1 for path in raw_images_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".png")
            if raw_images_dir.is_dir()
            else 0
        )
        expected_pngs = int(config["study"]["expected_unique_pngs"])
        if available_pngs < expected_pngs:
            raise FileNotFoundError(
                f"Full 25-class construction requires at least {expected_pngs:,} canonical PNG files, but only "
                f"{available_pngs:,} were found under {raw_images_dir}. The smaller public raw-image snapshot is "
                "insufficient for the canonical 5,033-case experiment."
            )
    source = pd.read_parquet(parquet_path)
    case_id_column = _first_column(source, CASE_ID_CANDIDATES)
    if case_id_column is None:
        raise ValueError(f"No case ID column found in {parquet_path}")

    case_manifest = pd.DataFrame({"case_id": source[case_id_column].astype(str)})
    for label in labels:
        source_column = _label_column(source, label)
        if source_column is None:
            raise ValueError(f"Missing target column for {label!r} in {parquet_path}")
        case_manifest[f"is_{slugify(label)}"] = pd.to_numeric(source[source_column], errors="coerce").fillna(0).gt(0).astype(np.uint8)

    label_columns = [f"is_{slugify(label)}" for label in labels]
    case_manifest["target_label_count"] = case_manifest[label_columns].sum(axis=1).astype(int)
    case_manifest["target_positive"] = case_manifest["target_label_count"].gt(0)

    legacy_cases = pd.read_csv(legacy_cases_path)
    legacy_case_id = _first_column(legacy_cases, CASE_ID_CANDIDATES)
    if legacy_case_id:
        legacy_cases[legacy_case_id] = legacy_cases[legacy_case_id].astype(str)
        metadata_columns = [
            column
            for column in legacy_cases.columns
            if column not in case_manifest.columns and column != legacy_case_id and str(column).lower() != "split"
        ]
        if metadata_columns:
            case_manifest = case_manifest.merge(
                legacy_cases[[legacy_case_id, *metadata_columns]].rename(columns={legacy_case_id: "case_id"}),
                on="case_id",
                how="left",
                validate="one_to_one",
            )

    raw_metadata_path = path_from(config, "raw_metadata_csv")
    if raw_metadata_path.is_file():
        raw_metadata = pd.read_csv(raw_metadata_path)
        raw_case_id = _first_column(raw_metadata, CASE_ID_CANDIDATES)
        if raw_case_id:
            raw_metadata[raw_case_id] = raw_metadata[raw_case_id].astype(str)
            portable_columns = [
                column
                for column in raw_metadata.columns
                if column != raw_case_id
                and column not in {"file_name", "image_id", "dataset_split"}
                and column not in case_manifest.columns
            ]
            if portable_columns:
                compact_metadata = raw_metadata.groupby(raw_case_id, as_index=False)[portable_columns].first()
                case_manifest = case_manifest.merge(
                    compact_metadata.rename(columns={raw_case_id: "case_id"}),
                    on="case_id",
                    how="left",
                    validate="one_to_one",
                )

    legacy_images = pd.read_csv(legacy_images_path)
    image_case_id = _first_column(legacy_images, CASE_ID_CANDIDATES)
    path_column = _first_column(legacy_images, PATH_CANDIDATES)
    raw_root = path_from(config, "raw_dataset_root")
    raw_images_dir = path_from(config, "raw_images_dir")
    rows: list[dict[str, Any]] = []

    if image_case_id and path_column:
        for order, row in legacy_images.reset_index(drop=True).iterrows():
            raw_path = row[path_column]
            resolved = resolve_image_path(raw_path, raw_root, raw_images_dir)
            rows.append(
                {
                    "case_id": str(row[image_case_id]),
                    "image_id": str(row.get("image_id", Path(str(raw_path)).stem)),
                    "image_order": int(row.get("image_order", row.get("slot", order))),
                    "source_image_path": str(raw_path),
                    "resolved_image_path": str(resolved),
                    "file_exists": resolved.is_file(),
                }
            )
    else:
        case_paths_column = next((column for column in source.columns if "image" in str(column).lower() and "path" in str(column).lower()), None)
        if case_paths_column is None:
            raise ValueError("Neither the legacy image manifest nor the case table exposes image paths")
        for _, row in source.iterrows():
            for order, raw_path in enumerate(_parse_sequence(row[case_paths_column])):
                resolved = resolve_image_path(raw_path, raw_root, raw_images_dir)
                rows.append(
                    {
                        "case_id": str(row[case_id_column]),
                        "image_id": Path(str(raw_path)).stem,
                        "image_order": order,
                        "source_image_path": str(raw_path),
                        "resolved_image_path": str(resolved),
                        "file_exists": resolved.is_file(),
                    }
                )

    image_manifest = pd.DataFrame(rows)
    image_manifest = image_manifest[image_manifest["case_id"].isin(set(case_manifest["case_id"]))].copy()
    image_manifest = image_manifest.sort_values(["case_id", "image_order", "image_id"], kind="stable").reset_index(drop=True)
    image_manifest["bag_slot"] = image_manifest.groupby("case_id").cumcount()
    image_manifest["retained_for_bag"] = image_manifest["bag_slot"].lt(int(config["study"]["max_images_per_case"]))

    if run_mode == "smoke":
        selected_ids = select_smoke_cases(case_manifest, image_manifest, labels, config["smoke"])
        case_manifest = case_manifest[case_manifest["case_id"].isin(selected_ids)].copy()
        image_manifest = image_manifest[image_manifest["case_id"].isin(selected_ids)].copy()

    case_counts = image_manifest.groupby("case_id").agg(
        manifest_image_count=("image_id", "size"),
        existing_image_count=("file_exists", "sum"),
        retained_image_count=("retained_for_bag", "sum"),
    )
    case_manifest = case_manifest.merge(case_counts, left_on="case_id", right_index=True, how="left")
    for column in ("manifest_image_count", "existing_image_count", "retained_image_count"):
        case_manifest[column] = case_manifest[column].fillna(0).astype(int)
    case_manifest["run_mode"] = run_mode
    image_manifest["run_mode"] = run_mode

    # Smoke and full runs intentionally use identical canonical filenames. A full
    # run replaces smoke artifacts only after it has built its complete outputs.
    case_output = path_from(config, "case_manifest")
    image_output = path_from(config, "image_manifest")
    write_csv(case_output, case_manifest)
    write_csv(image_output, image_manifest)
    write_json(
        case_output.with_suffix(".metadata.json"),
        {
            "run_mode": run_mode,
            "case_count": len(case_manifest),
            "image_row_count": len(image_manifest),
            "labels": labels,
            "complete": True,
        },
    )
    return case_manifest, image_manifest


def select_smoke_cases(
    cases: pd.DataFrame,
    images: pd.DataFrame,
    labels: list[str],
    smoke_config: dict[str, Any],
) -> set[str]:
    label_columns = [f"is_{slugify(label)}" for label in labels]
    required = int(smoke_config.get("samples_per_class", 2))
    selected: set[str] = set()
    coverage = {column: 0 for column in label_columns}

    # A smoke fixture must be executable with the images currently present on
    # the machine. The full manifest deliberately retains missing-image rows so
    # the audit can report them, but selecting those rows for a bounded smoke
    # run makes every downstream extractor fail for an unrelated data-transfer
    # reason. Restrict smoke candidates to cases whose retained bag images all
    # exist, while preserving the complete cohort behavior in full mode.
    retained = images[images["retained_for_bag"]].copy()
    eligible = retained.groupby("case_id")["file_exists"].agg(["all", "size"])
    eligible_ids = set(eligible.index[eligible["all"] & eligible["size"].gt(0)].astype(str))
    eligible_cases = cases[cases["case_id"].astype(str).isin(eligible_ids)].copy()
    if eligible_cases.empty:
        raise ValueError("No cases with complete retained image bags are available for the smoke fixture")

    ordered = eligible_cases.assign(
        _rarity=eligible_cases[label_columns]
        .mul(1 / eligible_cases[label_columns].sum().clip(lower=1), axis=1)
        .sum(axis=1)
    )
    ordered = ordered.sort_values(["_rarity", "case_id"], ascending=[False, True], kind="stable")
    while min(coverage.values(), default=required) < required:
        candidates = []
        for _, row in ordered.iterrows():
            case_id = str(row["case_id"])
            if case_id in selected:
                continue
            gain = sum(int(row[column] > 0 and coverage[column] < required) for column in label_columns)
            if gain:
                candidates.append((gain, case_id, row))
        if not candidates:
            missing = [column for column, count in coverage.items() if count < required]
            raise ValueError(f"Cannot construct smoke fixture with requested coverage: {missing}")
        _, case_id, row = max(candidates, key=lambda item: (item[0], item[1]))
        selected.add(case_id)
        for column in label_columns:
            coverage[column] += int(row[column] > 0)

    zero_count = int(smoke_config.get("all_zero_cases", 5))
    zero_ids = (
        eligible_cases.loc[eligible_cases[label_columns].sum(axis=1).eq(0), "case_id"]
        .astype(str)
        .sort_values()
        .head(zero_count)
    )
    selected.update(zero_ids)

    bag_target = int(smoke_config.get("cases_per_bag_length", 3))
    bag_lengths = retained[retained["case_id"].astype(str).isin(eligible_ids)].groupby("case_id").size().clip(upper=3)
    for bag_length in (1, 2, 3):
        ids = bag_lengths[bag_lengths.eq(bag_length)].index.astype(str)
        selected.update(sorted(set(ids) - selected)[:bag_target])
    return selected
