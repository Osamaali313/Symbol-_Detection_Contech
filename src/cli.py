#!/usr/bin/env python3
"""
Symbol matcher CLI.

Example:
    python -m src.cli \
        --pdf fixtures/drawings.pdf \
        --source-page 0 \
        --bbox 4824,1064,180,180 \
        --scope plan_type \
        --output outputs/

Reads a PDF, runs matching for the bbox you draw on `--source-page`, and
writes annotated PNGs plus a JSON results file to `--output`.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

# Allow running from project root: `python -m src.cli` or `python src/cli.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pages import load_pages_from_pdf, filter_pages_by_scope
from src.matcher import match_symbol_across_pages
from src.visualize import annotate_page, crop_match
import cv2


def parse_bbox(s: str) -> tuple[int, int, int, int]:
    parts = s.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be x,y,w,h")
    return tuple(int(p) for p in parts)  # type: ignore


def main():
    ap = argparse.ArgumentParser(description="Construction drawing symbol matcher")
    ap.add_argument("--pdf", required=True, help="Path to drawings PDF")
    ap.add_argument("--source-page", type=int, required=True,
                    help="Zero-indexed page containing the reference symbol")
    ap.add_argument("--bbox", type=parse_bbox, required=True,
                    help="Reference box: x,y,w,h in PIXEL coords at the rasterization DPI")
    ap.add_argument("--scope", choices=["page", "plan_type", "page_type"],
                    default="plan_type",
                    help="Which pages to search across")
    ap.add_argument("--dpi", type=int, default=200,
                    help="Rasterization DPI (default 200)")
    ap.add_argument("--cache-dir", default="fixtures",
                    help="Where to cache rasterized PNGs")
    ap.add_argument("--output", default="outputs",
                    help="Output directory")
    ap.add_argument("--stage1-threshold", type=float, default=0.45)
    ap.add_argument("--stage2-threshold", type=float, default=0.55)
    ap.add_argument("--nms-iou", type=float, default=0.30)
    ap.add_argument("--save-crops", action="store_true",
                    help="Also save a small crop of each match (the 'capture' record)")
    args = ap.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading pages from {args.pdf} at {args.dpi} DPI ...")
    t0 = time.time()
    pages = load_pages_from_pdf(args.pdf, cache_dir=args.cache_dir, dpi=args.dpi)
    print(f"      Loaded {len(pages)} pages in {time.time() - t0:.1f}s")
    for p in pages:
        print(f"      - page {p.page_index}: sheet_ref={p.sheet_ref!r} "
              f"page_type={p.page_type!r} name={p.page_name!r}")

    if not 0 <= args.source_page < len(pages):
        sys.exit(f"--source-page {args.source_page} out of range (0..{len(pages)-1})")
    source = pages[args.source_page]

    print(f"[2/4] Filtering pages with scope={args.scope!r} from source "
          f"{source.sheet_ref or source.page_id} ...")
    candidate_pages = filter_pages_by_scope(pages, source, args.scope)
    print(f"      {len(candidate_pages)} page(s) in scope: "
          f"{[p.sheet_ref or p.page_id for p in candidate_pages]}")

    print(f"[3/4] Running matcher (bbox={args.bbox} on page {args.source_page}) ...")
    t0 = time.time()
    matches = match_symbol_across_pages(
        candidate_pages, source, args.bbox,
        stage1_threshold=args.stage1_threshold,
        stage2_threshold=args.stage2_threshold,
        nms_iou=args.nms_iou,
    )
    elapsed = time.time() - t0
    print(f"      Found {len(matches)} match(es) in {elapsed:.1f}s "
          f"({elapsed / max(1, len(candidate_pages)):.1f}s/page)")

    print(f"[4/4] Writing outputs to {output_dir}/ ...")
    # JSON results
    results = {
        "source": {
            "pdf": str(args.pdf),
            "page_index": source.page_index,
            "sheet_ref": source.sheet_ref,
            "page_name": source.page_name,
            "page_type": source.page_type,
            "bbox": list(args.bbox),
        },
        "scope": args.scope,
        "config": {
            "dpi": args.dpi,
            "stage1_threshold": args.stage1_threshold,
            "stage2_threshold": args.stage2_threshold,
            "nms_iou": args.nms_iou,
        },
        "elapsed_seconds": round(elapsed, 2),
        "matches": [m.to_dict() for m in matches],
    }
    results_path = output_dir / "matches.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"      JSON: {results_path}")

    # Annotated PNGs (one per page that has matches)
    matches_by_page = {}
    for m in matches:
        matches_by_page.setdefault(m.page_index, []).append(m)
    for page in candidate_pages:
        pms = matches_by_page.get(page.page_index, [])
        qbbox = args.bbox if page.page_index == source.page_index else None
        # Always render the source page, even if no other matches there
        if not pms and qbbox is None:
            continue
        out = output_dir / f"annotated_{page.sheet_ref or f'page{page.page_index}'}.png"
        annotate_page(page.image, pms, out, query_bbox=qbbox)
        print(f"      PNG:  {out}  ({len(pms)} match(es))")

    # Per-match crops (the `captures` records)
    if args.save_crops:
        crops_dir = output_dir / "captures"
        crops_dir.mkdir(exist_ok=True)
        for i, m in enumerate(matches):
            page = pages[m.page_index]
            crop = crop_match(page.image, m.bbox)
            p = crops_dir / f"match_{i:04d}_{m.sheet_ref or m.page_id}.png"
            cv2.imwrite(str(p), crop)
        print(f"      Crops: {crops_dir}/ ({len(matches)} file(s))")

    # Pretty summary
    print()
    print(f"Top {min(10, len(matches))} matches:")
    print(f"  {'sheet':<10} {'page_type':<14} {'bbox':<22} {'rot':<5} {'score':<6} (s1/s2)")
    for m in matches[:10]:
        bbox_s = f"({m.bbox[0]},{m.bbox[1]},{m.bbox[2]},{m.bbox[3]})"
        print(f"  {(m.sheet_ref or m.page_id):<10} {(m.page_type or '-'):<14} "
              f"{bbox_s:<22} {m.rotation:<5} {m.score:.2f}  "
              f"({m.stage1_score:.2f}/{m.stage2_score:.2f})")


if __name__ == "__main__":
    main()
