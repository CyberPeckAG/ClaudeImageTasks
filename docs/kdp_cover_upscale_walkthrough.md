# Walkthrough: `kdp_cover_upscale.py`

Notes for explaining the script: what it does, how it is put together, and why
each choice was made. Line numbers refer to `kdp_cover_upscale.py` (439 lines).

## 1. The problem in one paragraph

AI-generated cover art usually comes out around 1024-1254 px square. Amazon KDP
needs a print cover at 300 DPI with 0.125 in of bleed. For an 8.5 x 8.5 in book
that is 2588 x 2625 px, so the art has to be enlarged about 2.5x. A plain resize
only makes the pixels bigger, so the result looks soft. This script uses an AI
super-resolution model (Real-ESRGAN) to redraw edges and fine detail at the
higher resolution. It also does the KDP geometry (trim, bleed, spine width) and
writes files KDP will accept.

```bash
python kdp_cover_upscale.py dragons.png --target print-front --trim 8.5x8.5 --sharpen 30
```

## 2. The pipeline

```
load_rgb ──> fit_to_aspect ──> upscale ──> (sharpen) ──> (assemble wrap) ──> save_outputs ──> report
 colour        crop / pad       AI or        optional       back + spine       JPEG / PDF        checks and
 + rotation    to target ratio  Lanczos      unsharp mask   + front            + PNG             warnings
```

`main()` (line 369) reads top to bottom in exactly this order, which makes it the
best place to start a walkthrough.

## 3. Section by section

### Constants (lines 35-67): the KDP rules in one place

`EBOOK_SIZE`, `BLEED_IN`, `SPINE_PER_PAGE`, the file-size limits and the model
URLs are all module-level constants.

**Why:** the business rules come from KDP's documentation and could change. Keeping
them together and named means there are no magic numbers in the logic, and an
update is a one-line edit.

### `Spec` dataclass and `build_spec()` (lines 71-133): geometry first

`build_spec()` turns the command-line options into a `Spec`: the output width and
height in pixels, the DPI, and, for wraps, where the front panel and spine sit.

- **Front cover:** trim width + 1 bleed (the spine edge gets none) by trim height + 2 bleeds.
- **Full wrap:** `2 × (trim + bleed) + spine`, where spine = pages × paper thickness.
- `px()` (line 94) rounds *up* (`math.ceil`) so the output is never a fraction of
  a pixel short of the minimum. The `- 1e-9` stops a float like `2550.0000001`
  being rounded up to 2551.

**Why:** working out the target size before touching any pixels keeps the
measurement logic separate from the image processing. It is also easy to test,
and a bad option (such as `--pages` missing for a wrap) fails before any slow work starts.

### `load_rgb()` (line 139): normalise the input

The function:

1. Applies the EXIF orientation, so phone photos aren't sideways.
2. Flattens transparency onto white. PNGs with alpha would otherwise turn black in JPEG.
3. Converts from any embedded ICC colour profile to sRGB.
4. Always returns 8-bit RGB.

**Why:** every later step can then assume one image format. Input mistakes such
as transparency, CMYK and rotation are dealt with once, at the start.

### `fit_to_aspect()` (line 165): different shapes

The source is rarely the target's exact shape. There are three strategies:

- `crop` (the default): trims the overflow. `--gravity` chooses which part to keep.
- `pad`: extends the canvas with a blurred, darkened copy of the art. This is the
  usual "letterbox" fix and looks better than flat bars.
- `stretch`: distorts the image. It's included for completeness and rarely wanted.

**Why:** the fit happens *before* upscaling, while the image is still small. That
makes it cheaper, and the AI model then works only on pixels that will appear in the output.

### The upscaling engines (lines 197-303): the core

**Optional dependency.** `ai_available()` tries to import `torch` and `spandrel`.
If they're missing, `--engine auto` falls back to Lanczos instead of crashing, so
the script still runs on a machine with only Pillow and NumPy installed.

**`RealESRGAN` class.** The model is loaded with `spandrel`, which reads the
weights file and builds the correct network. The class uses the GPU if one is
present and switches to half precision there.

**Tiled inference (`upscale`, line 238).** This is the most interesting code to explain:

- The image is processed in 192 px tiles, so memory stays flat for any input size.
- Each tile is read with 16 px of extra context on every side. The whole image is
  first mirror-padded (`np.pad(..., mode="reflect")`) so edge tiles have context too.
- The model's output for that extra margin is thrown away. Only the centre of each
  tile is written into the output array. This prevents visible seams: a network
  behaves badly near the edge of its input, so you keep only the part it saw in full.
- Pillow image → NumPy float array → PyTorch tensor (HWC → CHW plus a batch dimension)
  → model → back again. That conversion chain is worth being able to describe.

**`upscale()` wrapper (line 272).**

- If the source is already big enough, it simply downsamples.
- Real-ESRGAN is fixed at 4x, so the script oversamples and then reduces to the
  exact size with Lanczos. Super-resolving and then shrinking gives a sharper
  result than a smaller enlargement would.
- It runs a second pass only when 4x isn't enough, with a limit of two passes.
- `--strength` blends the AI result with a plain resize. That's a simple way to
  tone it down if the model invents too much detail.

**`lanczos_upscale()` (line 262).** The fallback enlarges in steps of at most 2x,
with light sharpening after each step. That keeps edges cleaner than one large jump.

### Saving and reporting (lines 309-366)

- **eBook:** JPEG at quality 95 with no chroma subsampling, so colour edges such
  as red text stay crisp. If the file goes over KDP's limit, the quality is
  lowered in steps.
- **Print:** a PDF (the file KDP actually takes) plus a lossless PNG master. Both
  carry 300 DPI metadata.
- **`report()`:** prints the sizes, the enlargement factor and the method used.
  It warns about KDP rules a script can't enforce, such as the barcode area and
  the minimum page count for spine text.

**Why report and not silently succeed?** For print work, a person must check the
result. The report tells them exactly what to look at.

### `main()` (line 369): the command-line interface

The CLI uses `argparse` with `choices=` to validate options, a custom type
function (`parse_trim`) that converts and validates "6x9" in one step, and the
module docstring as the `--help` text. `main(argv=None)` returns an exit code and
is called through `sys.exit(main())`. That pattern means the function can also be
called from tests with a list of arguments.

## 4. Python features on show

| Feature | Where |
| --- | --- |
| `@dataclass` | `Spec`, line 71 |
| Type hints, including `X \| None` and `tuple[...]` | throughout |
| `pathlib.Path` instead of string paths | throughout |
| Walrus operator `:=` | download loop, line 207 |
| Optional imports and graceful fallback | `ai_available()`, line 212 |
| Context managers | `urllib` download, `torch.inference_mode()` |
| NumPy slicing and padding | tiled inference, line 238 |
| `argparse` with custom types and choices | `parse_trim`, `main()` |
| Atomic file write (`.part` then rename) | `download_model()`, line 197 |

## 5. Honest weaknesses and what I'd improve

Being able to criticise the code is usually worth as much as explaining it.

1. **No automated tests.** `build_spec()` and `px()` are pure functions and the
   obvious first candidates for `pytest`, for example checking that 8.5 x 8.5 in
   gives 2588 x 2625 px.
2. **Downloaded model weights aren't checksum-verified.** A SHA-256 check against a
   known hash would guard against corrupt or tampered files. The weights are a
   PyTorch pickle, so this matters.
3. **`assert` is used for input validation** in the `--gravity` parsing. Asserts
   are removed when Python runs with `-O`, so it should raise a normal error instead.
4. **Tiles are hard-cut, not blended.** The context margin hides seams well in
   practice. Feathered blending of overlapping tiles would be more robust on
   smooth gradients.
5. **CPU speed:** about four minutes for a 1254 px source with the large model.
   The `general` model is several times faster, and a GPU is used automatically when present.
6. **`args` is passed around** to `upscale()`, `save_outputs()` and `report()`.
   Passing only the values each function needs would make them easier to test on
   their own.
7. **One real bug was found and fixed during testing.** An output name like
   `flight_front_8.5x8.5` was cut down to `flight_front_8` because
   `Path.with_suffix()` treated `.5x8` as a file extension. The fix only strips
   real image extensions. It's a good example of why a test run with realistic inputs matters.

## 6. Likely questions

- **"Why not just resize?"** Resizing only interpolates between existing pixels, so
  edges get softer. Super-resolution is trained to predict the detail a
  higher-resolution version would have. The side-by-side comparison crops show the
  difference clearly.
- **"Is the AI inventing detail?"** Somewhat, yes. Hence `--strength`, and the
  advice to check text and faces at 100% zoom. Line art and lettering come out
  well. Very fine textures can come out slightly "painted".
- **"Why 300 DPI and 0.125 in bleed?"** Both are KDP print requirements. The bleed
  lets the printer trim without leaving white slivers at the edges.
- **"How does it handle huge images?"** Tiling keeps the model's memory use flat.
  `Image.MAX_IMAGE_PIXELS` is raised because full print wraps exceed Pillow's
  default safety limit.
