from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import matplotlib as mpl


@contextmanager
def manuscript_style(style: dict) -> Iterator[None]:
    settings = {
        "font.family": style.get("font_family", "DejaVu Sans"),
        "figure.facecolor": style.get("background", "#FFFFFF"),
        "axes.facecolor": style.get("background", "#FFFFFF"),
        "axes.edgecolor": style.get("text", "#172033"),
        "axes.labelcolor": style.get("text", "#172033"),
        "text.color": style.get("text", "#172033"),
        "xtick.color": style.get("text", "#172033"),
        "ytick.color": style.get("text", "#172033"),
        "axes.grid": False,
        "savefig.dpi": style.get("dpi", 300),
        "savefig.bbox": "tight",
    }
    with mpl.rc_context(settings):
        yield
