#!/usr/bin/env python3
"""
Smoke test. Runs the canonical demo against fixtures/drawings.pdf and
asserts the expected outputs exist. Run this after install to verify
everything works.

    python tests/smoke_test.py
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "smoke"


def main():
    print("Running smoke test ...")
    OUT.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "src.cli",
        "--pdf", "fixtures/drawings.pdf",
        "--source-page", "0",
        "--bbox", "2641,1875,59,41",
        "--scope", "page",
        "--output", str(OUT.relative_to(ROOT)),
        "--dpi", "200",
    ]
    print("$ " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    assert result.returncode == 0, "CLI exited non-zero"

    # Assert outputs exist
    matches_json = OUT / "matches.json"
    annotated = OUT / "annotated_P-120.png"
    assert matches_json.exists(), f"missing {matches_json}"
    assert annotated.exists(), f"missing {annotated}"

    # Assert match content is sane
    with matches_json.open() as f:
        results = json.load(f)
    n = len(results["matches"])
    print(f"\nGot {n} matches.")
    assert n > 20, f"expected at least 20 matches on dense plumbing page, got {n}"
    top = results["matches"][0]
    assert top["score"] > 0.9, f"top match score should be > 0.9, got {top['score']}"
    assert top["sheet_ref"] == "P-120"
    assert top["page_type"] == "Plumbing"

    print("\n*** SMOKE TEST PASSED ***")
    print(f"Top match: sheet={top['sheet_ref']} bbox={top['bbox']} score={top['score']:.2f}")


if __name__ == "__main__":
    main()
