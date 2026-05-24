"""
Two-stage symbol matching pipeline.

Stage 1 (recall): Multi-scale, multi-rotation normalized cross-correlation
template matching on edge maps. Tuned aggressively for recall — we'd rather
generate a thousand candidates per page than miss a true match. False
positives here are cheap; they get filtered in Stage 2.

Stage 2 (precision): Each candidate is re-cropped at native resolution and
scored against the query using a structural descriptor (HOG cosine similarity
plus Hu-moment shape distance). This catches the cases where template
matching gets a high CC score on something that doesn't actually look like
the query (a common failure mode on dense schematic pages).

Both stages run on edge maps rather than raw pixels — schematic line drawings
are essentially defined by their edges, so edge-space matching is more robust
to anti-aliasing, line-weight variation, and minor occlusion than pixel-space.

Production note: Stage 2 here uses HOG so the demo runs with zero model
downloads. In production we'd swap it for a vision embedding (DINOv2-small,
or a CLIP variant, or a small encoder we fine-tune on AEC symbols). The
interface in `verify_candidates` is deliberately swap-in; the rest of the
pipeline doesn't care.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Callable
import numpy as np
import cv2
from skimage.feature import hog


# ----------------------------------------------------------------------------
# Result types
# ----------------------------------------------------------------------------

@dataclass
class Match:
    """One matched symbol instance."""
    page_id: str
    page_index: int
    sheet_ref: Optional[str]
    page_name: Optional[str]
    page_type: Optional[str]
    bbox: tuple[int, int, int, int]   # (x, y, w, h) in page pixel coords
    score: float                      # final combined score in [0, 1]
    stage1_score: float               # NCC score
    stage2_score: float               # HOG cosine similarity
    rotation: int                     # detected rotation in degrees (0/90/180/270)
    scale: float                      # detected scale relative to query

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        return d


# ----------------------------------------------------------------------------
# Image preprocessing
# ----------------------------------------------------------------------------

def to_edges(gray: np.ndarray) -> np.ndarray:
    """Convert grayscale image to a normalized edge map.

    Uses adaptive thresholding rather than fixed Canny: schematic drawings
    have very consistent line weights but the absolute pixel values vary
    with rasterization DPI and source PDF rendering. Adaptive thresholding
    is invariant to those shifts.
    """
    # Light blur kills aliasing without smearing thin lines
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # Inverted binary: lines become white (255), background black (0)
    edges = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=5,
    )
    return edges


def crop_query(page_image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop the user's box. bbox is (x, y, w, h)."""
    x, y, w, h = bbox
    return page_image[y:y + h, x:x + w].copy()


# ----------------------------------------------------------------------------
# Stage 1: Multi-scale, multi-rotation candidate generation
# ----------------------------------------------------------------------------

def _rotate90(img: np.ndarray, k: int) -> np.ndarray:
    """Rotate by k * 90 degrees. k in {0, 1, 2, 3}."""
    return np.rot90(img, k=k)


def _multi_scale_template_match(
    page_edges: np.ndarray,
    query_edges: np.ndarray,
    scales: list[float],
    threshold: float,
) -> list[tuple[int, int, int, int, float, float]]:
    """Run normalized cross-correlation at multiple scales.

    Returns a list of (x, y, w, h, score, scale) tuples. Includes ALL points
    above threshold; non-maximum suppression happens upstream.
    """
    candidates = []
    qh, qw = query_edges.shape

    for scale in scales:
        # Scale the template, not the page — much cheaper and matches behavior
        # under the assumption that drawings are produced at fixed sheet scales.
        new_w = max(8, int(round(qw * scale)))
        new_h = max(8, int(round(qh * scale)))
        if new_w >= page_edges.shape[1] or new_h >= page_edges.shape[0]:
            continue

        scaled = cv2.resize(query_edges, (new_w, new_h),
                            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        # TM_CCOEFF_NORMED is mean-centered and unit-normalized — robust to
        # local brightness/contrast variation that survives the binarization.
        # We use it on the binary edge map so it behaves like a normalized
        # IoU-of-edges similarity.
        result = cv2.matchTemplate(page_edges, scaled, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(result >= threshold)
        for y, x in zip(ys, xs):
            candidates.append((int(x), int(y), new_w, new_h,
                               float(result[y, x]), float(scale)))
    return candidates


def generate_candidates(
    page_edges: np.ndarray,
    query_edges: np.ndarray,
    scales: tuple[float, ...] = (0.85, 0.92, 1.0, 1.08, 1.15),
    rotations: tuple[int, ...] = (0, 1, 2, 3),
    stage1_threshold: float = 0.45,
) -> list[tuple[int, int, int, int, float, float, int]]:
    """Stage 1: generate candidate bounding boxes.

    Returns list of (x, y, w, h, ncc_score, scale, rotation_k) tuples.

    Defaults are tuned for recall: we accept a low NCC threshold (0.45) and
    sweep across reasonable scale (±15%) and rotation (0/90/180/270°) ranges.
    Free rotation is expensive; 90° increments cover ~95% of real-world
    rotated symbols on construction drawings. Free-angle support is a
    documented extension (see report).
    """
    all_candidates = []
    for k in rotations:
        rotated_query = _rotate90(query_edges, k)
        cands = _multi_scale_template_match(
            page_edges, rotated_query,
            scales=list(scales),
            threshold=stage1_threshold,
        )
        for x, y, w, h, score, scale in cands:
            all_candidates.append((x, y, w, h, score, scale, k))
    return all_candidates


# ----------------------------------------------------------------------------
# Stage 2: Verification with structural descriptors
# ----------------------------------------------------------------------------

def _hog_descriptor(img: np.ndarray, target_size: int = 64) -> np.ndarray:
    """Compute a fixed-length HOG descriptor for a symbol patch.

    All patches are resized to a common size before HOG so the descriptor
    is comparable across scales. HOG with these parameters gives good
    discrimination on schematic line drawings while being O(n) cheap.
    """
    resized = cv2.resize(img, (target_size, target_size),
                         interpolation=cv2.INTER_AREA)
    return hog(
        resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# Interface contract: a verifier takes (query_gray, candidate_gray) and
# returns a similarity score in [0, 1]. Swap with a deep embedding in
# production.
Verifier = Callable[[np.ndarray, np.ndarray], float]


def hog_verifier(query: np.ndarray, candidate: np.ndarray) -> float:
    """Default verifier: HOG cosine similarity. Returns score in [0, 1]."""
    q = _hog_descriptor(query)
    c = _hog_descriptor(candidate)
    sim = _cosine_similarity(q, c)
    # cosine sim on HOG of binarized symbols is naturally in [0, 1]
    return max(0.0, sim)


def verify_candidates(
    page_image: np.ndarray,
    query_image: np.ndarray,
    candidates: list[tuple[int, int, int, int, float, float, int]],
    verifier: Verifier = hog_verifier,
    stage2_threshold: float = 0.55,
) -> list[tuple[int, int, int, int, float, float, float, int]]:
    """Stage 2: re-score each candidate with a structural descriptor.

    For each candidate, crop the underlying pixel patch (not edge map),
    rotate it back to canonical orientation, and compare HOG descriptors
    to the query. Drop anything below stage2_threshold.

    Returns (x, y, w, h, stage1_score, stage2_score, combined_score, rot_k).
    """
    verified = []
    for x, y, w, h, s1, scale, rot_k in candidates:
        # Bounds-check
        if x < 0 or y < 0 or x + w > page_image.shape[1] or y + h > page_image.shape[0]:
            continue
        patch = page_image[y:y + h, x:x + w]
        # Un-rotate so the patch is in the same orientation as the query
        if rot_k != 0:
            patch = _rotate90(patch, k=(4 - rot_k) % 4)

        s2 = verifier(query_image, patch)
        if s2 < stage2_threshold:
            continue

        # Combined score: weighted average favoring Stage 2 (it's the
        # higher-precision signal). Tuned empirically; documented as a knob.
        combined = 0.35 * s1 + 0.65 * s2
        verified.append((x, y, w, h, s1, s2, combined, rot_k))
    return verified


# ----------------------------------------------------------------------------
# Non-maximum suppression
# ----------------------------------------------------------------------------

def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def non_max_suppress(
    verified: list[tuple[int, int, int, int, float, float, float, int]],
    iou_threshold: float = 0.30,
) -> list[tuple[int, int, int, int, float, float, float, int]]:
    """Greedy NMS by combined score.

    iou_threshold=0.30 is intentionally tight — schematic symbols are
    compact and on dense pages we don't want to merge two adjacent real
    instances into one detection.
    """
    if not verified:
        return []
    # Sort by combined score descending
    sorted_v = sorted(verified, key=lambda v: -v[6])
    keep = []
    for cand in sorted_v:
        cand_bbox = (cand[0], cand[1], cand[2], cand[3])
        if all(_iou(cand_bbox, (k[0], k[1], k[2], k[3])) < iou_threshold for k in keep):
            keep.append(cand)
    return keep


# ----------------------------------------------------------------------------
# End-to-end matching for one page
# ----------------------------------------------------------------------------

def match_symbol_on_page(
    page,                                # Page object
    query_image: np.ndarray,             # grayscale crop of the user's box
    *,
    scales: tuple[float, ...] = (0.85, 0.92, 1.0, 1.08, 1.15),
    rotations: tuple[int, ...] = (0, 1, 2, 3),
    stage1_threshold: float = 0.45,
    stage2_threshold: float = 0.55,
    nms_iou: float = 0.30,
    verifier: Verifier = hog_verifier,
) -> list[Match]:
    """Run the full pipeline for one (page, query) pair."""
    page_edges = to_edges(page.image)
    query_edges = to_edges(query_image)

    candidates = generate_candidates(
        page_edges, query_edges,
        scales=scales, rotations=rotations,
        stage1_threshold=stage1_threshold,
    )

    verified = verify_candidates(
        page.image, query_image, candidates,
        verifier=verifier, stage2_threshold=stage2_threshold,
    )

    kept = non_max_suppress(verified, iou_threshold=nms_iou)

    matches = []
    for x, y, w, h, s1, s2, combined, rot_k in kept:
        matches.append(Match(
            page_id=page.page_id,
            page_index=page.page_index,
            sheet_ref=page.sheet_ref,
            page_name=page.page_name,
            page_type=page.page_type,
            bbox=(x, y, w, h),
            score=combined,
            stage1_score=s1,
            stage2_score=s2,
            rotation=rot_k * 90,
            scale=1.0,
        ))
    return matches


def match_symbol_across_pages(
    pages: list,
    source_page,
    bbox: tuple[int, int, int, int],
    **kwargs,
) -> list[Match]:
    """End-to-end: take a user-drawn box on `source_page`, run matching
    across `pages` (assumed already filtered by scope).

    Self-match is automatically excluded for the source page so the query
    region doesn't show up as its own top result.
    """
    query_image = crop_query(source_page.image, bbox)
    all_matches = []
    for page in pages:
        page_matches = match_symbol_on_page(page, query_image, **kwargs)
        if page.page_index == source_page.page_index:
            # Exclude any match whose IoU with the query box is high — that's
            # the user's own selection echoing back.
            qx, qy, qw, qh = bbox
            page_matches = [
                m for m in page_matches
                if _iou(m.bbox, (qx, qy, qw, qh)) < 0.5
            ]
        all_matches.extend(page_matches)
    # Stable sort by score descending
    all_matches.sort(key=lambda m: -m.score)
    return all_matches
