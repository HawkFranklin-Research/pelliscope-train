from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from hawk_derm.config import path_from
from hawk_derm.io import sha256_file, write_csv, write_json


def _inspect_image(record: dict[str, Any], decode: bool) -> dict[str, Any]:
    path = Path(record["resolved_image_path"])
    result = dict(record)
    result.update({"sha256": "", "byte_size": 0, "width": None, "height": None, "decode_ok": False, "error": ""})
    if not path.is_file():
        result["error"] = "missing"
        return result
    try:
        result["byte_size"] = path.stat().st_size
        result["sha256"] = sha256_file(path)
        if decode:
            with Image.open(path) as image:
                image.load()
                result["width"], result["height"] = image.size
                result["image_mode"] = image.mode
        result["decode_ok"] = True
    except Exception as error:  # image failures must be preserved in the ledger
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def audit_dataset(
    config: dict[str, Any],
    case_manifest: pd.DataFrame,
    image_manifest: pd.DataFrame,
    *,
    decode: bool = True,
    workers: int = 8,
    strict: bool = False,
) -> dict[str, Any]:
    records = image_manifest.to_dict(orient="records")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        audited = list(pool.map(lambda record: _inspect_image(record, decode), records))
    image_audit = pd.DataFrame(audited)
    audit_dir = path_from(config, "audit_dir")
    audit_dir.mkdir(parents=True, exist_ok=True)
    write_csv(audit_dir / "image_audit.csv", image_audit)

    duplicate_paths = image_audit[image_audit.duplicated("resolved_image_path", keep=False)].sort_values("resolved_image_path")
    duplicate_hashes = image_audit[
        image_audit["sha256"].ne("") & image_audit.duplicated("sha256", keep=False)
    ].sort_values("sha256")
    cross_case_hashes = duplicate_hashes.groupby("sha256").filter(lambda group: group["case_id"].nunique() > 1)
    missing = image_audit[~image_audit["decode_ok"]]
    write_csv(audit_dir / "duplicate_paths.csv", duplicate_paths)
    write_csv(audit_dir / "duplicate_hashes.csv", duplicate_hashes)
    write_csv(audit_dir / "cross_case_duplicate_hashes.csv", cross_case_hashes)
    write_csv(audit_dir / "missing_or_unreadable_images.csv", missing)

    raw_metadata_path = path_from(config, "raw_metadata_csv")
    reconciliation: dict[str, Any] = {}
    if raw_metadata_path.is_file():
        raw = pd.read_csv(raw_metadata_path)
        if "case_id" in raw:
            raw_case_ids = set(raw["case_id"].astype(str))
            canonical_case_ids = set(case_manifest["case_id"].astype(str))
            case_reconciliation = pd.DataFrame(
                [
                    *({"case_id": value, "status": "matched"} for value in sorted(raw_case_ids & canonical_case_ids)),
                    *({"case_id": value, "status": "raw_metadata_only"} for value in sorted(raw_case_ids - canonical_case_ids)),
                    *({"case_id": value, "status": "canonical_only"} for value in sorted(canonical_case_ids - raw_case_ids)),
                ]
            )
            write_csv(audit_dir / "source_case_reconciliation.csv", case_reconciliation)
            reconciliation.update(
                {
                    "raw_metadata_cases": len(raw_case_ids),
                    "matched_cases": len(raw_case_ids & canonical_case_ids),
                    "raw_metadata_only_cases": len(raw_case_ids - canonical_case_ids),
                    "canonical_only_cases": len(canonical_case_ids - raw_case_ids),
                }
            )
        if "file_name" in raw:
            raw_files = set(raw["file_name"].dropna().astype(str).map(lambda value: Path(value).name))
            canonical_files = set(image_manifest["resolved_image_path"].astype(str).map(lambda value: Path(value).name))
            image_reconciliation = pd.DataFrame(
                [
                    *({"file_name": value, "status": "matched"} for value in sorted(raw_files & canonical_files)),
                    *({"file_name": value, "status": "raw_metadata_only"} for value in sorted(raw_files - canonical_files)),
                    *({"file_name": value, "status": "canonical_only"} for value in sorted(canonical_files - raw_files)),
                ]
            )
            write_csv(audit_dir / "source_image_reconciliation.csv", image_reconciliation)
            reconciliation.update(
                {
                    "raw_metadata_image_rows": int(len(raw)),
                    "raw_metadata_unique_files": len(raw_files),
                    "matched_image_names": len(raw_files & canonical_files),
                    "raw_metadata_only_image_names": len(raw_files - canonical_files),
                    "canonical_only_image_names": len(canonical_files - raw_files),
                }
            )

    labels = [column for column in case_manifest.columns if column.startswith("is_")]
    summary = {
        "case_count": int(case_manifest["case_id"].nunique()),
        "manifest_image_rows": int(len(image_manifest)),
        "unique_resolved_paths": int(image_audit["resolved_image_path"].nunique()),
        "unique_decoded_hashes": int(image_audit.loc[image_audit["decode_ok"], "sha256"].nunique()),
        "decoded_images": int(image_audit["decode_ok"].sum()),
        "missing_or_unreadable_images": int((~image_audit["decode_ok"]).sum()),
        "duplicated_path_rows": int(len(duplicate_paths)),
        "duplicated_hash_rows": int(len(duplicate_hashes)),
        "cross_case_duplicate_hash_rows": int(len(cross_case_hashes)),
        "retained_image_rows": int(image_manifest["retained_for_bag"].sum()),
        "images_excluded_by_bag_limit": int((~image_manifest["retained_for_bag"]).sum()),
        "all_zero_target_cases": int(case_manifest[labels].sum(axis=1).eq(0).sum()),
        "single_label_cases": int(case_manifest[labels].sum(axis=1).eq(1).sum()),
        "multi_label_cases": int(case_manifest[labels].sum(axis=1).gt(1).sum()),
        "total_positive_labels": int(case_manifest[labels].to_numpy().sum()),
        "complete": True,
        **reconciliation,
    }
    expected = {
        "case_count": int(config["study"]["canonical_case_count"]),
        "manifest_image_rows": int(config["study"]["expected_manifest_image_rows"]),
        "unique_resolved_paths": int(config["study"]["expected_unique_pngs"]),
    }
    summary["expected"] = expected
    summary["expectations_met"] = all(summary[key] == value for key, value in expected.items())
    write_json(audit_dir / "data_audit_summary.json", summary)
    if strict and not summary["expectations_met"]:
        differences = {key: {"observed": summary[key], "expected": value} for key, value in expected.items() if summary[key] != value}
        raise ValueError(f"Canonical data audit failed: {differences}")
    return summary
