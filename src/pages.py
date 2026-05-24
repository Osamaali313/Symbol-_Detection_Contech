from __future__ import annotations
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import cv2


# Discipline prefix → page_type mapping. AIA standard sheet numbering.
DISCIPLINE_MAP = {
    "A": "Architectural",
    "S": "Structural",
    "M": "Mechanical",
    "P": "Plumbing",
    "E": "Electrical",
    "FP": "Fire Protection",
    "C": "Civil",
    "L": "Landscape",
    "T": "Telecom",
}


@dataclass
class Page:
    """One rasterized drawing sheet plus its metadata."""

    page_id: str                      # stable identifier (sheet ref or index)
    page_index: int                   # zero-indexed position in source PDF
    image_path: Path                  # path to rasterized PNG
    image: np.ndarray = field(repr=False)  # grayscale uint8
    sheet_ref: Optional[str] = None   # e.g. "P-120"
    page_name: Optional[str] = None   # e.g. "Plumbing Second Floor Construction Plan"
    page_type: Optional[str] = None   # e.g. "Plumbing"
    dpi: int = 200

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[:2]


def _extract_sheet_ref(pdf_path: Path, page_index: int) -> tuple[Optional[str], Optional[str]]:
    """Best-effort extraction of sheet number and plan name from page text.

    Looks for AIA-style sheet numbers (e.g. P-120, E-201, A-100) and common
    plan-name patterns. Returns (sheet_ref, page_name).
    """
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", "-f", str(page_index + 1),
             "-l", str(page_index + 1), str(pdf_path), "-"],
            capture_output=True, timeout=10
        )
        text = result.stdout.decode("utf-8", errors="replace")
    except Exception:
        return None, None

    # AIA-style sheet number: letter(s) + dash + digits, anchored as a token
    sheet_ref = None
    for m in re.finditer(r"\b([A-Z]{1,2})-?(\d{2,4}[A-Z]?)\b", text):
        prefix, num = m.group(1), m.group(2)
        if prefix in DISCIPLINE_MAP:
            sheet_ref = f"{prefix}-{num}"
            break

    # Plan-name heuristic: look for "<Discipline> ... PLAN" lines
    page_name = None
    for line in text.splitlines():
        s = line.strip()
        if 8 < len(s) < 120 and "PLAN" in s.upper() and any(
            d.upper() in s.upper() for d in DISCIPLINE_MAP.values()
        ):
            page_name = s
            break

    return sheet_ref, page_name


def _infer_page_type(sheet_ref: Optional[str]) -> Optional[str]:
    if not sheet_ref:
        return None
    m = re.match(r"^([A-Z]{1,2})-?\d", sheet_ref)
    if m:
        return DISCIPLINE_MAP.get(m.group(1))
    return None


def load_pages_from_pdf(
    pdf_path: str | Path,
    cache_dir: str | Path,
    dpi: int = 200,
) -> list[Page]:
    """Rasterize a PDF to PNG pages at the given DPI and load them as Page objects.

    Caches rasterized pages on disk so re-runs are instant.
    """
    pdf_path = Path(pdf_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    stem = pdf_path.stem
    prefix = cache_dir / f"{stem}_{dpi}dpi"

    existing = sorted(cache_dir.glob(f"{stem}_{dpi}dpi-*.png"))
    if not existing:
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), str(pdf_path), str(prefix)],
            check=True
        )
        existing = sorted(cache_dir.glob(f"{stem}_{dpi}dpi-*.png"))

    pages = []
    for i, img_path in enumerate(existing):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        sheet_ref, page_name = _extract_sheet_ref(pdf_path, i)
        page_type = _infer_page_type(sheet_ref)
        pages.append(Page(
            page_id=sheet_ref or f"page_{i}",
            page_index=i,
            image_path=img_path,
            image=img,
            sheet_ref=sheet_ref,
            page_name=page_name,
            page_type=page_type,
            dpi=dpi,
        ))
    return pages


def filter_pages_by_scope(
    pages: list[Page],
    source_page: Page,
    scope: str,
) -> list[Page]:
    """Filter pages by the scope selected in the UI.

    scope ∈ {"page", "plan_type", "page_type"}
    - "page"      : just this page
    - "plan_type" : pages with similar plan name (e.g. all Floor Plans of any level)
    - "page_type" : all pages with the same discipline (Plumbing, Electrical, etc.)
    """
    if scope == "page":
        return [source_page]

    if scope == "page_type":
        if not source_page.page_type:
            return [source_page]
        return [p for p in pages if p.page_type == source_page.page_type]

    if scope == "plan_type":
        # Simple heuristic: same discipline AND a shared "plan-type token" in name
        # ("Construction Plan", "Power Plan", "Lighting Plan", etc.)
        # Strip floor-level words so "First Floor Power Plan" ~ "Second Floor Power Plan".
        def signature(name: Optional[str]) -> str:
            if not name:
                return ""
            s = name.upper()
            for word in ["FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH",
                         "BASEMENT", "GROUND", "ROOF", "1ST", "2ND", "3RD", "4TH",
                         "FLOOR", "LEVEL", "MEZZANINE", "PENTHOUSE"]:
                s = s.replace(word, "")
            return " ".join(s.split())
        sig = signature(source_page.page_name)
        if not sig:
            return [p for p in pages if p.page_type == source_page.page_type]
        return [p for p in pages if signature(p.page_name) == sig]

    raise ValueError(f"Unknown scope: {scope}")
