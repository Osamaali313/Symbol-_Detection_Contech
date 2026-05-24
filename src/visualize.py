"""Render match results back onto page images for visual inspection."""

from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np


def annotate_page(
    page_image: np.ndarray,
    matches: list,
    output_path: str | Path,
    query_bbox: tuple[int, int, int, int] | None = None,
    max_label_score: float = 0.999,
) -> Path:
    """Draw bounding boxes for matches on a copy of the page.

    The query box (if provided) is drawn in red so it's distinguishable
    from detected matches (drawn in green).
    """
    # Convert grayscale to BGR for color annotation
    if page_image.ndim == 2:
        canvas = cv2.cvtColor(page_image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = page_image.copy()

    # Query box in red
    if query_bbox is not None:
        x, y, w, h = query_bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 4)
        cv2.putText(canvas, "QUERY", (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Matches in green, ordered worst-first so best draws on top
    for m in sorted(matches, key=lambda m: m.score):
        x, y, w, h = m.bbox
        # Color intensity scales with score so reviewers eyeball confidence
        green = int(255 * min(1.0, m.score / max_label_score))
        color = (0, green, 0)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 3)
        label = f"{m.score:.2f}"
        cv2.putText(canvas, label, (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return output_path


def crop_match(page_image: np.ndarray, bbox: tuple[int, int, int, int],
               padding: int = 8) -> np.ndarray:
    """Return a padded crop around a match — used for `captures` records
    that the upstream system stores per the spec."""
    x, y, w, h = bbox
    H, W = page_image.shape[:2]
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(W, x + w + padding)
    y1 = min(H, y + h + padding)
    return page_image[y0:y1, x0:x1].copy()
