#!/usr/bin/env python3
"""Upscale a low-resolution image to Amazon KDP cover specifications.

Targets
  ebook        Kindle eBook cover: 1600 x 2560 px JPEG (KDP's ideal size).
  print-front  Front cover only, with bleed, at 300 DPI (for KDP Cover Creator
               or for dropping into a wrap layout yourself).
  print-wrap   Full print-on-demand paperback wrap (back + spine + front) with
               bleed at 300 DPI, saved as a print-ready PDF.

Upscaling uses Real-ESRGAN (AI super-resolution) when torch + spandrel are
installed, and falls back to stepped Lanczos resampling with sharpening.

Examples
  python kdp_cover_upscale.py art.png --target ebook
  python kdp_cover_upscale.py art.png --target print-front --trim 6x9
  python kdp_cover_upscale.py art.png --target print-wrap --trim 6x9 \
      --pages 240 --paper cream --layout front-only
"""
from __future__ import annotations

import argparse
import io
import math
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageFilter, ImageOps

Image.MAX_IMAGE_PIXELS = 400_000_000

# --------------------------------------------------------------------------
# KDP specifications
# --------------------------------------------------------------------------
EBOOK_SIZE = (1600, 2560)  # width, height: KDP's ideal eBook cover
PRINT_DPI = 300
BLEED_IN = 0.125
# Spine width per page, inches (KDP paperback).
SPINE_PER_PAGE = {
    "white": 0.002252,
    "cream": 0.0025,
    "color-standard": 0.002252,
    "color-premium": 0.002347,
}
MIN_PAGES_FOR_SPINE_TEXT = 79
MAX_FILE_BYTES_EBOOK = 50 * 1024 * 1024
MAX_FILE_BYTES_PRINT = 650 * 1024 * 1024

MODELS = {
    # name: (url, description)
    "x4plus": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "best quality for photos / painted art (slow on CPU)",
    ),
    "general": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "compact general model (much faster, slightly softer)",
    ),
    "anime": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "flat-colour illustration, line art, cartoons",
    ),
}
CACHE_DIR = Path.home() / ".cache" / "kdp-cover-upscale"


@dataclass
class Spec:
    """Pixel geometry of the file to produce."""

    width: int
    height: int
    dpi: int
    description: str
    front_box: tuple[int, int, int, int] | None = None  # for front-only wraps
    spine_px: int = 0


def parse_trim(text: str) -> tuple[float, float]:
    try:
        w, h = (float(p) for p in text.lower().replace("×", "x").split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"trim must look like 6x9, got {text!r}")
    if not (4 <= w <= 8.5 and 6 <= h <= 11.69):
        raise argparse.ArgumentTypeError(
            f"trim {w}x{h} in is outside KDP paperback sizes (4-8.5 x 6-11.69 in)"
        )
    return w, h


def px(inches: float, dpi: int) -> int:
    return math.ceil(inches * dpi - 1e-9)


def build_spec(args) -> Spec:
    if args.target == "ebook":
        w, h = EBOOK_SIZE
        return Spec(w, h, 300, "Kindle eBook cover (1.6:1)")

    tw, th = args.trim
    dpi, bleed = args.dpi, BLEED_IN
    if args.target == "print-front":
        # Bleed on top, bottom and the outside edge; the spine edge has none.
        return Spec(
            px(tw + bleed, dpi),
            px(th + 2 * bleed, dpi),
            dpi,
            f'paperback front cover, {tw:g}x{th:g}" trim + bleed',
        )

    if args.pages is None:
        raise SystemExit("--pages is required for --target print-wrap (sets spine width)")
    if not 24 <= args.pages <= 828:
        raise SystemExit("KDP paperbacks must have 24-828 pages")
    spine_in = args.pages * SPINE_PER_PAGE[args.paper]
    spine_px = round(spine_in * dpi)
    side_px = px(tw + bleed, dpi)  # one cover panel + its outside bleed
    width = 2 * side_px + spine_px
    height = px(th + 2 * bleed, dpi)
    spec = Spec(
        width,
        height,
        dpi,
        f'paperback wrap, {tw:g}x{th:g}" trim, {args.pages} pp {args.paper} '
        f'(spine {spine_in:.3f}")',
        spine_px=spine_px,
    )
    if args.layout == "front-only":
        spec.front_box = (side_px + spine_px, 0, width, height)
    return spec


# --------------------------------------------------------------------------
# Loading and colour handling
# --------------------------------------------------------------------------
def load_rgb(path: Path) -> Image.Image:
    """Open an image as 8-bit sRGB, honouring EXIF rotation and ICC profiles."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    icc = img.info.get("icc_profile")
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, rgba).convert("RGB")
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            dst = ImageCms.createProfile("sRGB")
            if img.mode not in ("RGB", "CMYK", "L"):
                img = img.convert("RGB")
            img = ImageCms.profileToProfile(img, src, dst, outputMode="RGB")
        except ImageCms.PyCMSError:
            pass
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# --------------------------------------------------------------------------
# Fitting the source to the target aspect ratio
# --------------------------------------------------------------------------
def fit_to_aspect(img: Image.Image, tw: int, th: int, fit: str, gravity: tuple[float, float]):
    """Return the source re-shaped to the target aspect ratio (still low-res)."""
    sw, sh = img.size
    target = tw / th
    if abs(sw / sh - target) < 1e-3 or fit == "stretch":
        return img
    if fit == "crop":
        if sw / sh > target:  # too wide
            nw = round(sh * target)
            x = round((sw - nw) * gravity[0])
            return img.crop((x, 0, x + nw, sh))
        nh = round(sw / target)
        y = round((sh - nh) * gravity[1])
        return img.crop((0, y, sw, y + nh))
    # pad: extend the canvas with a blurred, darkened copy of the artwork.
    if sw / sh > target:
        cw, ch = sw, round(sw / target)
    else:
        cw, ch = round(sh * target), sh
    cover = max(cw / sw, ch / sh)
    bg = img.resize((math.ceil(sw * cover), math.ceil(sh * cover)), Image.LANCZOS)
    bx, by = (bg.width - cw) // 2, (bg.height - ch) // 2
    bg = bg.crop((bx, by, bx + cw, by + ch))
    bg = bg.filter(ImageFilter.GaussianBlur(max(cw, ch) / 25))
    bg = Image.eval(bg, lambda v: int(v * 0.8))
    bg.paste(img, (round((cw - sw) * gravity[0]), round((ch - sh) * gravity[1])))
    return bg


# --------------------------------------------------------------------------
# Upscaling engines
# --------------------------------------------------------------------------
def download_model(name: str) -> Path:
    url = MODELS[name][0]
    dest = CACHE_DIR / url.rsplit("/", 1)[1]
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  downloading model weights: {url}")
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    tmp.rename(dest)
    return dest


def ai_available() -> bool:
    try:
        import spandrel  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


class RealESRGAN:
    """Tiled Real-ESRGAN 4x inference (GPU if present, otherwise CPU)."""

    def __init__(self, model_name: str, tile: int = 192, overlap: int = 16):
        import torch
        from spandrel import ModelLoader

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        descriptor = ModelLoader(device=self.device).load_from_file(str(download_model(model_name)))
        self.model = descriptor.model.eval()
        self.scale = descriptor.scale
        self.half = self.device == "cuda"
        if self.half:
            self.model.half()
        self.tile, self.overlap = tile, overlap

    def upscale(self, img: Image.Image) -> Image.Image:
        torch = self.torch
        arr = np.asarray(img, dtype=np.float32) / 255.0
        h, w, _ = arr.shape
        s, pad, step = self.scale, self.overlap, self.tile
        # Reflect-pad so every tile carries real context past its edges.
        padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        out = np.empty((h * s, w * s, 3), dtype=np.float32)
        tiles = [(y, x) for y in range(0, h, step) for x in range(0, w, step)]
        for i, (y, x) in enumerate(tiles, 1):
            y2, x2 = min(y + step, h), min(x + step, w)
            patch = padded[y : y2 + 2 * pad, x : x2 + 2 * pad]
            t = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(self.device)
            if self.half:
                t = t.half()
            with torch.inference_mode():
                r = self.model(t).float().clamp_(0, 1)
            r = r[0].permute(1, 2, 0).cpu().numpy()
            out[y * s : y2 * s, x * s : x2 * s] = r[pad * s : (pad + y2 - y) * s, pad * s : (pad + x2 - x) * s]
            print(f"\r  upscaling tile {i}/{len(tiles)} on {self.device}", end="", flush=True)
        print()
        return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8))


def lanczos_upscale(img: Image.Image, factor: float) -> Image.Image:
    """Stepped Lanczos enlargement: <=2x per step with light sharpening."""
    while factor > 1.0001:
        step = min(2.0, factor)
        img = img.resize((round(img.width * step), round(img.height * step)), Image.LANCZOS)
        img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=45, threshold=2))
        factor /= step
    return img


def upscale(img: Image.Image, tw: int, th: int, args) -> tuple[Image.Image, str]:
    """Bring `img` (already at target aspect) to exactly tw x th pixels."""
    factor = max(tw / img.width, th / img.height)
    if factor <= 1.0:
        return img.resize((tw, th), Image.LANCZOS), "downsample (source already large enough)"

    engine = args.engine
    if engine == "auto":
        engine = "realesrgan" if ai_available() else "lanczos"
        if engine == "lanczos":
            print("  note: torch/spandrel not installed; using Lanczos fallback "
                  "(pip install -r requirements.txt for AI upscaling)")
    if engine == "realesrgan":
        if not ai_available():
            raise SystemExit("--engine realesrgan needs: pip install torch spandrel")
        net = RealESRGAN(args.model, tile=args.tile)
        result, passes = img, 0
        while max(tw / result.width, th / result.height) > 1.0 and passes < 2:
            result = net.upscale(result)
            passes += 1
        used = f"Real-ESRGAN {args.model} x{net.scale}" + (f" ({passes} passes)" if passes > 1 else "")
        if args.strength < 1.0:
            # Blend back towards plain resampling to tame over-smoothing/invented detail.
            plain = img.resize(result.size, Image.LANCZOS)
            result = Image.blend(plain, result, args.strength)
            used += f", strength {args.strength:g}"
    else:
        result = lanczos_upscale(img, factor)
        used = "stepped Lanczos"
    if result.size != (tw, th):
        result = result.resize((tw, th), Image.LANCZOS)
    return result, used


# --------------------------------------------------------------------------
# Wrap assembly, saving, reporting
# --------------------------------------------------------------------------
def parse_color(text: str | None, img: Image.Image) -> tuple[int, int, int]:
    if text:
        text = text.lstrip("#")
        return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    # Default: average colour of the front cover's spine-side edge.
    strip = np.asarray(img.crop((0, 0, max(1, img.width // 50), img.height)))
    return tuple(int(v) for v in strip.reshape(-1, 3).mean(axis=0))  # type: ignore[return-value]


def save_outputs(img: Image.Image, spec: Spec, args, stem: Path) -> list[Path]:
    dpi = (spec.dpi, spec.dpi)
    written = []
    if args.target == "ebook":
        out = stem.with_name(stem.name + ".jpg")
        quality = 95
        while True:
            img.save(out, "JPEG", quality=quality, subsampling=0, optimize=True, dpi=dpi)
            if out.stat().st_size <= MAX_FILE_BYTES_EBOOK or quality <= 70:
                break
            quality -= 5
        written.append(out)
    else:
        pdf = stem.with_name(stem.name + ".pdf")
        img.save(pdf, "PDF", resolution=float(spec.dpi), quality=95, subsampling=0)
        written.append(pdf)
        png = stem.with_name(stem.name + ".png")
        img.save(png, "PNG", dpi=dpi, optimize=False)
        written.append(png)
    return written


def report(src: Image.Image, src_path: Path, spec: Spec, method: str, factor: float, files: list[Path], args):
    print("\nResult")
    print(f"  source        {src_path.name}  {src.width} x {src.height} px")
    print(f"  target        {spec.description}")
    print(f"  output size   {spec.width} x {spec.height} px @ {spec.dpi} DPI "
          f'({spec.width / spec.dpi:.3f} x {spec.height / spec.dpi:.3f} in)')
    print(f"  enlargement   {factor:.2f}x via {method}")
    for f in files:
        print(f"  wrote         {f}  ({f.stat().st_size / 1e6:.1f} MB)")

    warnings = []
    if factor > 4:
        warnings.append(f"{factor:.1f}x enlargement is large; inspect fine detail and any text at 100% zoom.")
    limit = MAX_FILE_BYTES_EBOOK if args.target == "ebook" else MAX_FILE_BYTES_PRINT
    for f in files:
        if f.suffix in (".jpg", ".pdf") and f.stat().st_size > limit:
            warnings.append(f"{f.name} exceeds KDP's upload size limit.")
    if args.target == "print-wrap":
        if args.pages < MIN_PAGES_FOR_SPINE_TEXT:
            warnings.append(f"KDP does not allow spine text under {MIN_PAGES_FOR_SPINE_TEXT} pages.")
        warnings.append("Keep text/logos 0.25 in from trim edges and 0.0625 in from spine folds; "
                        "leave the barcode area (2 x 1.2 in, lower right of the back cover) clear.")
    if args.target == "print-front":
        warnings.append("KDP print uploads need a full wrap PDF; use this file in Cover Creator "
                        "or run --target print-wrap.")
    for w in warnings:
        print(f"  ! {w}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="source image (PNG, JPEG, TIFF, WebP ...)")
    p.add_argument("-o", "--output", type=Path, help="output path stem (extension is chosen for you)")
    p.add_argument("--target", choices=["ebook", "print-front", "print-wrap"], default="ebook")
    p.add_argument("--trim", type=parse_trim, default=(6.0, 9.0), help="print trim size in inches, e.g. 6x9")
    p.add_argument("--pages", type=int, help="page count (print-wrap; sets spine width)")
    p.add_argument("--paper", choices=sorted(SPINE_PER_PAGE), default="white")
    p.add_argument("--layout", choices=["full", "front-only"], default="full",
                   help="print-wrap: image is the whole wrap, or just the front (back/spine get a solid colour)")
    p.add_argument("--back-color", help="hex colour for back + spine with --layout front-only (default: sampled)")
    p.add_argument("--dpi", type=int, default=PRINT_DPI, help="print resolution (KDP requires >= 300)")
    p.add_argument("--fit", choices=["crop", "pad", "stretch"], default="crop",
                   help="how to reconcile a different aspect ratio (default: crop)")
    p.add_argument("--gravity", default="0.5,0.5",
                   help="crop/pad anchor as x,y in 0..1 (0.5,0.5 = centre; 0.5,0 = keep the top)")
    p.add_argument("--engine", choices=["auto", "realesrgan", "lanczos"], default="auto")
    p.add_argument("--model", choices=sorted(MODELS), default="x4plus",
                   help="; ".join(f"{k}: {v[1]}" for k, v in MODELS.items()))
    p.add_argument("--strength", type=float, default=1.0,
                   help="0..1 mix of AI result vs plain resampling (lower = more faithful, softer)")
    p.add_argument("--sharpen", type=float, default=0.0, help="final unsharp-mask amount in percent (e.g. 40)")
    p.add_argument("--tile", type=int, default=192, help="AI tile size in source pixels (lower uses less memory)")
    args = p.parse_args(argv)

    if args.dpi < 300 and args.target != "ebook":
        p.error("KDP print covers require at least 300 DPI")
    if not 0 <= args.strength <= 1:
        p.error("--strength must be between 0 and 1")
    try:
        gravity = tuple(min(1.0, max(0.0, float(v))) for v in args.gravity.split(","))
        assert len(gravity) == 2
    except (ValueError, AssertionError):
        p.error("--gravity must look like 0.5,0.5")

    spec = build_spec(args)
    src = load_rgb(args.input)
    print(f"Source {src.width} x {src.height} px -> {spec.description}: {spec.width} x {spec.height} px")

    if spec.front_box:
        x0, y0, x1, y1 = spec.front_box
        art_w, art_h = x1 - x0, y1 - y0
    else:
        art_w, art_h = spec.width, spec.height

    fitted = fit_to_aspect(src, art_w, art_h, args.fit, gravity)
    factor = max(art_w / fitted.width, art_h / fitted.height)
    art, method = upscale(fitted, art_w, art_h, args)
    if args.sharpen > 0:
        art = art.filter(ImageFilter.UnsharpMask(radius=1.5, percent=int(args.sharpen), threshold=2))

    if spec.front_box:
        final = Image.new("RGB", (spec.width, spec.height), parse_color(args.back_color, art))
        final.paste(art, spec.front_box[:2])
    else:
        final = art
    assert final.size == (spec.width, spec.height) and final.mode == "RGB"

    if args.output:
        known = args.output.suffix.lower() in (".jpg", ".jpeg", ".png", ".pdf", ".tif", ".tiff")
        stem = args.output.with_suffix("") if known else args.output
    else:
        stem = args.input.with_name(f"{args.input.stem}_kdp_{args.target}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    files = save_outputs(final, spec, args, stem)
    report(src, args.input, spec, method, factor, files, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
