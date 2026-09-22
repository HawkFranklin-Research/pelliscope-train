#!/usr/bin/env python3
"""
Package all experiment scores, predictions, metrics, tables, figures, and manifests into a single zip archive.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


def export_results_zip(
    repo_root: Path,
    output_zip: Path,
    include_weights: bool = False,
) -> None:
    output_zip = output_zip.resolve()
    print(f"Creating results archive: {output_zip}")

    included_count = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. Include all reports (tables, figures, statistics, evaluations, plot data, manifests)
        reports_dir = repo_root / "reports"
        if reports_dir.is_dir():
            for path in sorted(reports_dir.rglob("*")):
                if path.is_file() and not path.name.startswith("."):
                    rel = path.relative_to(repo_root)
                    zf.write(path, arcname=str(rel))
                    included_count += 1
                    total_uncompressed_bytes += path.stat().st_size

        # 2. Include all prediction, metric, threshold, and manifest files from artifacts/models
        models_dir = repo_root / "artifacts" / "models"
        if models_dir.is_dir():
            for path in sorted(models_dir.rglob("*")):
                if not path.is_file() or path.name.startswith("."):
                    continue
                # By default, include CSVs, JSONs, YAMLs
                # Exclude .pt (binary weights) unless requested
                if path.suffix.lower() in (".csv", ".json", ".yaml", ".yml", ".txt"):
                    rel = path.relative_to(repo_root)
                    zf.write(path, arcname=str(rel))
                    included_count += 1
                    total_uncompressed_bytes += path.stat().st_size
                elif include_weights and path.suffix.lower() in (".pt", ".pth", ".bin"):
                    rel = path.relative_to(repo_root)
                    zf.write(path, arcname=str(rel))
                    included_count += 1
                    total_uncompressed_bytes += path.stat().st_size

        # 3. Include feature metadata and manifests (excluding multi-gigabyte .npz feature banks)
        features_dir = repo_root / "artifacts" / "features"
        if features_dir.is_dir():
            for path in sorted(features_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in (".json", ".csv"):
                    rel = path.relative_to(repo_root)
                    zf.write(path, arcname=str(rel))
                    included_count += 1
                    total_uncompressed_bytes += path.stat().st_size

        # 4. Include canonical split and case manifests
        manifests_dir = repo_root / "data" / "manifests"
        if manifests_dir.is_dir():
            for path in sorted(manifests_dir.rglob("*.csv")):
                rel = path.relative_to(repo_root)
                zf.write(path, arcname=str(rel))
                included_count += 1
                total_uncompressed_bytes += path.stat().st_size

        splits_dir = repo_root / "data" / "splits"
        if splits_dir.is_dir():
            for path in sorted(splits_dir.rglob("*.csv")):
                rel = path.relative_to(repo_root)
                zf.write(path, arcname=str(rel))
                included_count += 1
                total_uncompressed_bytes += path.stat().st_size

    archive_size_mb = output_zip.stat().st_size / (1024 * 1024)
    uncompressed_mb = total_uncompressed_bytes / (1024 * 1024)
    print(f"Done! Packaged {included_count} files.")
    print(f"Uncompressed: {uncompressed_mb:.2f} MB -> Compressed Zip: {archive_size_mb:.2f} MB")
    print(f"Archive location: {output_zip}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export all results, predictions, scores, and tables to a zip file.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("pelliscope_results_and_scores.zip"),
        help="Path for output zip file (default: pelliscope_results_and_scores.zip)",
    )
    parser.add_argument(
        "--include-weights",
        action="store_true",
        help="Include .pt neural network weights in the archive (default: false)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root path",
    )
    args = parser.parse_args()

    export_results_zip(
        repo_root=args.repo_root,
        output_zip=args.output,
        include_weights=args.include_weights,
    )


if __name__ == "__main__":
    main()
