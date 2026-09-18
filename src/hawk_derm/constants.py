from __future__ import annotations

import re


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


CASE_ID_CANDIDATES = ("case_id", "caseid", "case", "submission_id", "scin_case_id")
IMAGE_ID_CANDIDATES = ("image_id", "imageid", "image", "image_name")
PATH_CANDIDATES = ("image_path", "path", "file_path", "filepath", "image")
SPLITS = ("train", "validation", "test")
