# ClaudeImageTasks

## kdp_cover_upscale.py

Uplifts a low-resolution image to Amazon KDP cover specifications. It uses
Real-ESRGAN super-resolution (reconstructs edges and detail rather than just
resizing), oversamples to 4x, then reduces to the exact pixel size required.

```bash
pip install -r requirements.txt
```

| Target | Output | Size |
| --- | --- | --- |
| `ebook` | JPEG | 1600 x 2560 px (KDP ideal, 1.6:1) |
| `print-front` | PDF + PNG | trim + 0.125 in bleed (top, bottom, outside edge) at 300 DPI |
| `print-wrap` | PDF + PNG | back + spine + front + bleed at 300 DPI; spine from `--pages` / `--paper` |

```bash
# Kindle eBook cover
python kdp_cover_upscale.py art.png --target ebook

# Square 8.5 x 8.5 in paperback front cover (e.g. for KDP Cover Creator)
python kdp_cover_upscale.py art.png --target print-front --trim 8.5x8.5 --sharpen 30

# Full paperback wrap built from a front-only image
python kdp_cover_upscale.py art.png --target print-wrap --trim 6x9 \
    --pages 240 --paper cream --layout front-only --back-color 1b1b2f
```

Useful options:

- `--model x4plus|general|anime` - best quality / fast / flat-colour line art.
- `--strength 0..1` - blend the AI result with plain resampling if it looks over-processed.
- `--sharpen N` - final unsharp mask, percent.
- `--fit crop|pad|stretch` and `--gravity x,y` - handle a source with a different aspect ratio.
- `--engine lanczos` - no AI, no extra dependencies.

Model weights are downloaded once to `~/.cache/kdp-cover-upscale`. On a 4-core
CPU a 1254 px square source takes about four minutes with `x4plus`.

`output/` holds covers produced with the script.
