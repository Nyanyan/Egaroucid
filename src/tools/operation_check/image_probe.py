"""保存した盤面PNGの大きさ・色・透過を検証する（Pillowが必要）。"""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def inspect_image(path, pixels, size=None, alphas=(), clipboard=None):
    with path.open("rb") as stream:
        image = Image.open(stream).convert("RGBA")
    result = {"file": str(path), "width": image.width, "height": image.height,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "pixels": [], "passed": True}
    if size and list(image.size) != size:
        result["passed"] = False
    for pixel in pixels:
        x, y, *expected = pixel
        actual = list(image.getpixel((x, y)))
        passed = actual[:len(expected)] == expected
        result["pixels"].append({"x": x, "y": y, "expected": expected, "actual": actual, "passed": passed})
        result["passed"] &= passed
    for x, y, expected in alphas:
        actual = image.getpixel((x, y))[3]
        result["pixels"].append({"x": x, "y": y, "expected_alpha": expected, "actual_alpha": actual, "passed": actual == expected})
        result["passed"] &= actual == expected
    if clipboard is not None:
        png, dib = clipboard.get("png", {}), clipboard.get("dibv5", {})
        matches = (png.get("sha256") == result["sha256"] and png.get("width") == image.width and
                   png.get("height") == image.height and dib.get("width") == image.width and
                   dib.get("height") == -image.height)
        result["clipboard_matches"] = matches
        result["passed"] &= matches
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--pixel", action="append", default=[], help="x,y,r,g,b または x,y,r,g,b,a")
    parser.add_argument("--size", type=int, nargs=2)
    parser.add_argument("--alpha", action="append", default=[], help="x,y,a")
    parser.add_argument("--clipboard-json", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    pixels = [[int(value) for value in pixel.split(",")] for pixel in args.pixel]
    alphas = [[int(value) for value in point.split(",")] for point in args.alpha]
    clipboard = json.loads(args.clipboard_json.read_text(encoding="utf-8")) if args.clipboard_json else None
    result = inspect_image(args.image, pixels, args.size, alphas, clipboard)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
