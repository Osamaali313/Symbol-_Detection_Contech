"""Capture Streamlit screenshots for README."""
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
    page.wait_for_timeout(5000)

    # Hide Streamlit deploy header for cleaner shots
    page.add_style_tag(content="""
        header[data-testid="stHeader"] { visibility: hidden !important; }
        [data-testid="stToolbar"] { display: none !important; }
    """)
    page.wait_for_timeout(500)

    page.screenshot(path=str(OUT / "01_hero.png"), full_page=False)
    print("saved 01_hero.png")

    page.screenshot(path=str(OUT / "02_full_page.png"), full_page=True)
    print("saved 02_full_page.png")

    sidebar = page.locator('[data-testid="stSidebar"]')
    try:
        sidebar.screenshot(path=str(OUT / "03_sidebar.png"))
        print("saved 03_sidebar.png")
    except Exception as e:
        print(f"sidebar shot failed: {e}")

    try:
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "04_scope_identify.png"), full_page=False)
        print("saved 04_scope_identify.png")
    except Exception as e:
        print(f"scope shot failed: {e}")

    browser.close()
print("done")
