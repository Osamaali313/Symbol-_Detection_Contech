# Symbol Matcher

End-to-end symbol matching for construction drawings. User draws a box around a
symbol on one page; system finds matching instances across selected pages.

Built around the spec in `Symbol_Matching_Quick_Def.pdf`. Validated on real
plumbing CDs (Kent Substance Abuse Center, sheets P-120 / P-121 / P-130).

## What it does

1. **Loads a multi-page PDF** of construction drawings and rasterizes each page
   at a configurable DPI (default 200).
2. **Extracts page metadata** automatically from each sheet — sheet reference
   (e.g. `P-120`), plan name (e.g. `PLUMBING SECOND FLOOR CONSTRUCTION PLAN`),
   and discipline (`Plumbing`, `Electrical`, etc.) — using AIA-standard sheet
   numbering conventions.
3. **Filters pages by scope** the way the spec defines:
   - `page` — just the source page
   - `plan_type` — all pages with the same plan type across floors
     (e.g. all "Construction Plan" pages, not "Above Ceiling" diagrams)
   - `page_type` — all pages of the same discipline
4. **Runs a two-stage match pipeline** (see [REPORT.md](docs/REPORT.md) for the
   architecture rationale):
   - Stage 1: multi-scale, multi-rotation normalized cross-correlation on edge
     maps. Tuned for high recall.
   - Stage 2: HOG-descriptor cosine-similarity verification. Tuned for
     precision. Pluggable interface — swap for a learned embedding (DINOv2,
     CLIP, fine-tuned encoder) in production with no other pipeline changes.
   - Non-maximum suppression with tight IoU threshold to preserve adjacent
     instances on dense pages.
5. **Outputs**:
   - `matches.json` — every match with `sheet_ref`, `page_name`, `page_type`,
     `bbox`, `score`, `rotation`, plus per-stage scores for debugging
   - `annotated_<sheet>.png` — each page with green boxes on matches, red on
     query, score label on every box
   - `captures/` — per-match cropped patches (the `captures` records that the
     spec asks the upstream system to persist)

## Setup

Requires Python 3.10+, poppler (`pdftoppm` and `pdftotext`).

```bash
# system deps
apt-get install -y poppler-utils    # or: brew install poppler

# python deps
pip install -r requirements.txt
```

`requirements.txt`:
```
opencv-python-headless>=4.8
numpy>=1.24
scikit-image>=0.21
pymupdf>=1.23      # for the future vector-PDF fast path; not used in v1
```

## Run

```bash
# Match a hexagon callout on page 0, search current page only
python -m src.cli \
    --pdf fixtures/drawings.pdf \
    --source-page 0 \
    --bbox 2641,1875,59,41 \
    --scope page \
    --output outputs/run1

# Same query, searched across all Plumbing pages
python -m src.cli \
    --pdf fixtures/drawings.pdf \
    --source-page 0 \
    --bbox 2641,1875,59,41 \
    --scope page_type \
    --output outputs/run2

# Floor drain symbol (bonus: non-text small symbol)
python -m src.cli \
    --pdf fixtures/drawings.pdf \
    --source-page 2 \
    --bbox 3001,2266,19,20 \
    --scope page_type \
    --output outputs/run3 \
    --stage2-threshold 0.50

# Save per-match crops as 'capture' records
python -m src.cli ... --save-crops
```

The `bbox` is `x,y,w,h` in pixel coordinates at the rasterization DPI. In a real
UI the user would drag a rectangle on the page; this CLI accepts those pixel
coords directly so it's drop-in behind any browser/desktop frontend that knows
where the user clicked.

## Tuning knobs

| Flag | Default | What it does |
|---|---|---|
| `--dpi` | 200 | Rasterization DPI. Higher = more accurate, slower, more memory. 200 is the sweet spot for typical 24x36" sheets. |
| `--stage1-threshold` | 0.45 | NCC threshold for candidate generation. Lower = more candidates = higher recall = slower. |
| `--stage2-threshold` | 0.55 | HOG similarity threshold for verification. Lower = more matches kept = higher recall, more false positives. |
| `--nms-iou` | 0.30 | Non-max suppression IoU. Lower = stricter = fewer overlapping detections. |

For the spec's "minimize false negatives" requirement, the default config is
already tuned recall-first. If you want to tighten precision (e.g. for a
final-export pass), raise `--stage2-threshold` to 0.70.

## Project layout

```
src/
  pages.py       PDF loading, page metadata extraction, scope filtering
  matcher.py     Two-stage matching pipeline (the core algorithm)
  visualize.py   Annotation and capture-crop rendering
  cli.py         CLI entry point
fixtures/
  drawings.pdf   The 3-page plumbing CD set used for validation
outputs/         Run outputs (gitignored in a real repo)
docs/
  REPORT.md      Technical report — approach, tradeoffs, scaling design
```

## Validation runs

The fixture PDF (`fixtures/drawings.pdf`) contains real plumbing CDs.
Three runs are demonstrated:

| Run | Query | Scope | Pages searched | Matches | Notes |
|---|---|---|---|---|---|
| Hexagon callout (`P14`) | (2641,1875,59,41) on P-120 | page | 1 | 143 | Score gradient: P14-text → 0.95+, other-text hexagons → 0.75-0.90 |
| Same | same | page_type | 3 | 186 | Correctly distributes across P-120, P-121; correctly returns 0 on kitchen page (different symbol set) |
| Floor drain ⊘ | (3001,2266,19,20) on P-130 | page_type | 3 | 34 | Mix of 0° and 180° rotations; bonus / non-text symbol |

End-to-end runtime on the 3-page set, single-threaded Python at 200 DPI:
~57 seconds. See REPORT.md for the production scaling design.

## What's not in v1

- Free-angle rotation (only 0/90/180/270°). See report for the
  Fourier-Mellin / log-polar approach for production.
- Vector-PDF fast path (extract drawing primitives instead of rasterizing).
  Stubbed via the `pymupdf` dependency; implementation in the report.
- Learned embedding in Stage 2 (DINOv2 / CLIP / fine-tuned encoder). The
  `Verifier` callable interface in `matcher.py` is the swap point.
- UI. The CLI returns coordinates and annotated PNGs; a real UI would render
  the user's rectangle drag, send it to the API, and display the matches
  back. See report for the architecture diagram.

See [docs/REPORT.md](docs/REPORT.md) for the full technical write-up.
