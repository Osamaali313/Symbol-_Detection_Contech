"""Downscale annotated PNGs and crop detail regions for README."""
import cv2
from pathlib import Path

OUT = Path("docs/screenshots")
TARGET_W = 1600

# Downscale full-page annotated images
for name in ["result_hex_p120.png", "result_floordrain_p130.png"]:
    p = OUT / name
    img = cv2.imread(str(p))
    h, w = img.shape[:2]
    scale = TARGET_W / w
    small = cv2.resize(img, (TARGET_W, int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(p), small, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"{name}: {w}x{h} -> {TARGET_W}x{int(h*scale)}")

# Downscale query preview
p = OUT / "preview_query.png"
img = cv2.imread(str(p))
h, w = img.shape[:2]
if w > 600:
    scale = 600 / w
    img = cv2.resize(img, (600, int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(p), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"preview_query: -> 600x{int(h*scale)}")

# Crop detail region around the hexagon callouts on P-120 (native coords)
src = cv2.imread(str(Path("outputs/run_hex_plantype/annotated_P-120.png")))
# Region with many hexagons around (2400, 1800) to (4500, 3700)
y1, y2, x1, x2 = 1750, 3700, 2400, 4500
crop = src[y1:y2, x1:x2]
ch, cw = crop.shape[:2]
target_w = 1400
scale = target_w / cw
crop = cv2.resize(crop, (target_w, int(ch * scale)), interpolation=cv2.INTER_AREA)
cv2.imwrite(str(OUT / "detail_hex_matches.png"), crop, [cv2.IMWRITE_PNG_COMPRESSION, 9])
print(f"detail_hex_matches: {target_w}x{int(ch*scale)}")

# Floor drain detail crop on P-130
src = cv2.imread(str(Path("outputs/run_floordrain/annotated_P-130.png")))
h, w = src.shape[:2]
y1, y2, x1, x2 = 1800, 3500, 2400, 4400
crop = src[y1:y2, x1:x2]
ch, cw = crop.shape[:2]
target_w = 1400
scale = target_w / cw
crop = cv2.resize(crop, (target_w, int(ch * scale)), interpolation=cv2.INTER_AREA)
cv2.imwrite(str(OUT / "detail_floordrain_matches.png"), crop, [cv2.IMWRITE_PNG_COMPRESSION, 9])
print(f"detail_floordrain_matches: {target_w}x{int(ch*scale)}")

print("done")
