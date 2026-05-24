"""Drive Streamlit UI: draw box, click identify, screenshot results."""
from playwright.sync_api import sync_playwright
from pathlib import Path

OUT = Path("docs/screenshots")
OUT.mkdir(parents=True, exist_ok=True)
URL = "http://localhost:8501"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000}, device_scale_factor=2)
    page = ctx.new_page()
    page.goto(URL, wait_until="networkidle")
    page.wait_for_timeout(6000)

    page.add_style_tag(content="""
        header[data-testid="stHeader"] { visibility: hidden !important; }
        [data-testid="stToolbar"] { display: none !important; }
    """)

    # Select scope=page first (faster - 1 page only)
    try:
        page.get_by_text("This page only").click()
        page.wait_for_timeout(800)
    except Exception as e:
        print(f"scope click: {e}")

    # Find drawable canvas iframe
    canvas_frame = None
    for f in page.frames:
        if "component" in (f.url or "") or "drawable" in (f.url or "").lower():
            canvas_frame = f
            break
    if canvas_frame is None:
        for f in page.frames:
            if f != page.main_frame:
                canvas_frame = f
                break

    if canvas_frame:
        # Scroll the iframe into view from the parent page
        page.locator("iframe").first.scroll_into_view_if_needed()
        page.wait_for_timeout(500)
        canvas_frame.wait_for_selector("canvas", state="attached", timeout=10000)
        canvases = canvas_frame.locator("canvas")
        n = canvases.count()
        print(f"canvases in frame: {n}")
        target = canvases.nth(n - 1)
        try:
            target.scroll_into_view_if_needed(timeout=3000)
        except Exception:
            pass
        page.wait_for_timeout(500)
        box = target.bounding_box()
        print(f"canvas box: {box}")

        # Draw bbox 2641,1875,59,41 — but at the displayed scale of 1200x800
        scale_x = box["width"] / 7200
        scale_y = box["height"] / 4800
        x1 = box["x"] + 2641 * scale_x
        y1 = box["y"] + 1875 * scale_y
        x2 = x1 + 59 * scale_x
        y2 = y1 + 41 * scale_y
        # Make box at least 30x30 in screen pixels for visibility
        if (x2 - x1) < 30: x2 = x1 + 30
        if (y2 - y1) < 30: y2 = y1 + 30

        page.mouse.move(x1, y1)
        page.mouse.down()
        for i in range(1, 21):
            page.mouse.move(x1 + (x2 - x1) * i / 20, y1 + (y2 - y1) * i / 20)
            page.wait_for_timeout(15)
        page.mouse.up()
        page.wait_for_timeout(1500)

        # Screenshot with box drawn
        page.screenshot(path=str(OUT / "05_box_drawn.png"), full_page=False)
        print("saved 05_box_drawn.png")

    # Click identify
    try:
        identify = page.get_by_role("button", name="Identify matches")
        identify.click(timeout=5000)
        print("clicked identify, waiting for results...")
        page.wait_for_selector("text=Found", timeout=180000)
        page.wait_for_timeout(3000)
        page.screenshot(path=str(OUT / "06_results.png"), full_page=False)
        print("saved 06_results.png")
        page.screenshot(path=str(OUT / "07_results_full.png"), full_page=True)
        print("saved 07_results_full.png")
    except Exception as e:
        print(f"identify flow failed: {e}")
        page.screenshot(path=str(OUT / "error.png"), full_page=True)

    browser.close()
print("done")
