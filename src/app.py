from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Optional

# Make `src` importable when run as `streamlit run src/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.pages import load_pages_from_pdf, filter_pages_by_scope, Page
from src.matcher import match_symbol_across_pages
from src.visualize import annotate_page


# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------

st.set_page_config(page_title="Symbol Matcher", layout="wide")
st.title("Symbol Matcher")
st.caption(
    "Draw a box around a symbol on the page below, choose a scope, "
    "and click Identify. The matcher will find similar instances across "
    "the selected pages."
)


# ----------------------------------------------------------------------------
# Sidebar — PDF + tuning knobs
# ----------------------------------------------------------------------------

with st.sidebar:
    st.header("Document")
    pdf_path = st.text_input(
        "PDF path",
        value="fixtures/drawings.pdf",
        help="Path to the multi-page drawings PDF, relative to the project root.",
    )
    dpi = st.select_slider(
        "Rasterization DPI",
        options=[100, 150, 200, 300],
        value=200,
        help="Higher = more accurate but slower. 200 is the sweet spot.",
    )

    st.divider()
    st.header("Matching parameters")
    stage1_threshold = st.slider(
        "Stage 1 threshold (recall)",
        min_value=0.20, max_value=0.80, value=0.45, step=0.05,
        help="NCC threshold for candidate generation. Lower = more candidates.",
    )
    stage2_threshold = st.slider(
        "Stage 2 threshold (precision)",
        min_value=0.30, max_value=0.95, value=0.55, step=0.05,
        help="HOG similarity threshold for verification. Lower = keeps more matches.",
    )
    nms_iou = st.slider(
        "NMS IoU",
        min_value=0.10, max_value=0.60, value=0.30, step=0.05,
        help="Lower = stricter suppression of overlapping detections.",
    )

    st.divider()
    st.caption(
        "Tip: the **confidence slider** below the results lets you tighten "
        "the threshold without re-running the matcher."
    )


# ----------------------------------------------------------------------------
# Load PDF (cached so re-runs are fast)
# ----------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading and rasterizing PDF ...")
def _load_pages(pdf_path: str, dpi: int) -> list[Page]:
    return load_pages_from_pdf(pdf_path, cache_dir="fixtures", dpi=dpi)


if not Path(pdf_path).exists():
    st.error(f"PDF not found at `{pdf_path}`. Set the path in the sidebar.")
    st.stop()

pages = _load_pages(pdf_path, dpi)

if not pages:
    st.error("No pages found in PDF.")
    st.stop()


# ----------------------------------------------------------------------------
# Page picker
# ----------------------------------------------------------------------------

def _page_label(p: Page) -> str:
    bits = []
    if p.sheet_ref:
        bits.append(p.sheet_ref)
    if p.page_type:
        bits.append(p.page_type)
    if p.page_name:
        bits.append(p.page_name)
    return " — ".join(bits) or f"Page {p.page_index}"


page_idx = st.selectbox(
    "Source page (where the reference symbol lives)",
    options=range(len(pages)),
    format_func=lambda i: f"[{i}] {_page_label(pages[i])}",
)
source = pages[page_idx]


# ----------------------------------------------------------------------------
# Canvas — downsample the page so it fits in the browser, let user draw a box
# ----------------------------------------------------------------------------

st.subheader("1. Draw a box around a symbol")

H_native, W_native = source.shape
# Cap display width — anything bigger than ~1400px is awkward in a browser
DISPLAY_W = 1200
scale = DISPLAY_W / W_native
display_h = int(H_native * scale)

# Convert grayscale page to a PIL image for the canvas background
bg_image = Image.fromarray(source.image).convert("RGB")
bg_image_small = bg_image.resize((DISPLAY_W, display_h), Image.LANCZOS)

st.caption(
    f"Page is {W_native}x{H_native}px at {dpi} DPI; displayed at "
    f"{DISPLAY_W}x{display_h}px (scale factor {scale:.3f}). "
    f"The bbox you draw is rescaled to native resolution before matching."
)

canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.10)",
    stroke_width=2,
    stroke_color="#ff0000",
    background_image=bg_image_small,
    drawing_mode="rect",
    update_streamlit=True,
    height=display_h,
    width=DISPLAY_W,
    key=f"canvas_{page_idx}_{dpi}",
)

# Extract the most recent rectangle from the canvas
bbox_native: Optional[tuple[int, int, int, int]] = None
if canvas_result.json_data is not None:
    rects = [o for o in canvas_result.json_data.get("objects", [])
             if o.get("type") == "rect"]
    if rects:
        r = rects[-1]   # most recent box wins
        # st_canvas returns left/top/width/height in display coords
        dx, dy = float(r["left"]), float(r["top"])
        dw, dh = float(r["width"]), float(r["height"])
        if dw > 5 and dh > 5:   # ignore stray micro-clicks
            bbox_native = (
                int(round(dx / scale)),
                int(round(dy / scale)),
                int(round(dw / scale)),
                int(round(dh / scale)),
            )

if bbox_native:
    x, y, w, h = bbox_native
    st.success(f"Reference bbox (native pixels): x={x}, y={y}, w={w}, h={h}")
    # Show the cropped query so the user can verify what they boxed
    crop = source.image[y:y + h, x:x + w]
    if crop.size > 0:
        st.image(crop, caption="Selected reference symbol", width=200)
else:
    st.info("Draw a rectangle on the page above to define your reference symbol.")


# ----------------------------------------------------------------------------
# Scope picker
# ----------------------------------------------------------------------------

st.subheader("2. Choose a scope")
scope = st.radio(
    "Where should the matcher search?",
    options=["page", "plan_type", "page_type"],
    horizontal=True,
    format_func=lambda s: {
        "page": "This page only",
        "plan_type": "Pages with similar plan name (e.g. all Construction Plan floors)",
        "page_type": f"All {source.page_type or '—'} pages",
    }[s],
    index=1,
)

# Show which pages will be searched
candidate_pages = filter_pages_by_scope(pages, source, scope)
with st.expander(f"Pages in scope ({len(candidate_pages)})", expanded=False):
    for p in candidate_pages:
        st.markdown(f"- **{p.sheet_ref or p.page_id}** — {p.page_name or '(no name)'}")


# ----------------------------------------------------------------------------
# Identify button -> run the matcher
# ----------------------------------------------------------------------------

st.subheader("3. Identify")

if "matches" not in st.session_state:
    st.session_state.matches = None
    st.session_state.last_run_bbox = None
    st.session_state.last_run_scope = None
    st.session_state.elapsed = 0.0

run = st.button(
    "Identify matches",
    type="primary",
    disabled=(bbox_native is None),
)

if run and bbox_native:
    with st.spinner(
        f"Matching across {len(candidate_pages)} page(s). "
        f"Roughly 20s/page at 200 DPI ..."
    ):
        t0 = time.time()
        matches = match_symbol_across_pages(
            candidate_pages, source, bbox_native,
            stage1_threshold=stage1_threshold,
            stage2_threshold=stage2_threshold,
            nms_iou=nms_iou,
        )
        st.session_state.matches = matches
        st.session_state.last_run_bbox = bbox_native
        st.session_state.last_run_scope = scope
        st.session_state.elapsed = time.time() - t0


# ----------------------------------------------------------------------------
# Results — confidence slider, annotated previews, JSON download
# ----------------------------------------------------------------------------

if st.session_state.matches is not None:
    all_matches = st.session_state.matches
    st.success(
        f"Found {len(all_matches)} match(es) across "
        f"{len(set(m.page_index for m in all_matches))} page(s) in "
        f"{st.session_state.elapsed:.1f}s."
    )

    # Score distribution + confidence slider — lets the reviewer tighten
    # precision without retriggering the matcher.
    if all_matches:
        scores = np.array([m.score for m in all_matches])
        min_s, max_s = float(scores.min()), float(scores.max())
        col_a, col_b = st.columns([2, 3])
        with col_a:
            threshold = st.slider(
                "Confidence threshold",
                min_value=round(min_s, 2),
                max_value=round(max_s, 2),
                value=round(min_s, 2),
                step=0.01,
                help=(
                    "Filter matches by combined score. Move right for "
                    "high-precision exact-instance matches only; left for "
                    "broader symbol-class matches."
                ),
            )
        with col_b:
            st.bar_chart(
                np.histogram(scores, bins=20)[0],
                height=140,
            )
            st.caption("Score distribution across all matches")

        filtered = [m for m in all_matches if m.score >= threshold]
    else:
        filtered = []
        threshold = 0.0

    st.markdown(f"### Showing {len(filtered)} match(es) at threshold {threshold:.2f}")

    # Render annotated previews per page that has matches
    by_page: dict[int, list] = {}
    for m in filtered:
        by_page.setdefault(m.page_index, []).append(m)

    qbox = st.session_state.last_run_bbox

    for page in candidate_pages:
        pms = by_page.get(page.page_index, [])
        page_qbox = qbox if page.page_index == source.page_index else None
        if not pms and page_qbox is None:
            continue

        st.markdown(f"#### {page.sheet_ref or page.page_id} — {len(pms)} match(es)")
        st.caption(page.page_name or "")

        # Render to a temp PNG, then downsample for display
        # (full-resolution annotated PNGs are 30-60MB; the browser can't show
        # them at native size anyway)
        tmp_out = Path("outputs") / "_ui_cache"
        tmp_out.mkdir(parents=True, exist_ok=True)
        out_png = tmp_out / f"annotated_{page.sheet_ref or page.page_index}.png"
        annotate_page(page.image, pms, out_png, query_bbox=page_qbox)
        annotated = cv2.imread(str(out_png))
        # downscale for display
        disp = cv2.resize(annotated, (DISPLAY_W, int(annotated.shape[0] * scale)),
                          interpolation=cv2.INTER_AREA)
        st.image(disp, channels="BGR", use_column_width=True)

    # JSON download
    import json
    results = {
        "source": {
            "sheet_ref": source.sheet_ref,
            "page_name": source.page_name,
            "page_type": source.page_type,
            "page_index": source.page_index,
            "bbox": list(st.session_state.last_run_bbox or []),
        },
        "scope": st.session_state.last_run_scope,
        "threshold_displayed": threshold,
        "elapsed_seconds": round(st.session_state.elapsed, 2),
        "matches": [m.to_dict() for m in filtered],
    }
    st.download_button(
        "Download matches.json",
        data=json.dumps(results, indent=2),
        file_name="matches.json",
        mime="application/json",
    )

elif bbox_native is None:
    st.info("Draw a box and click Identify to run the matcher.")

# Footer
st.divider()
st.caption(
    "Two-stage hybrid matcher: edge-map NCC for recall, HOG cosine similarity "
    "for precision. See `docs/REPORT.md` for the full architecture."
)
