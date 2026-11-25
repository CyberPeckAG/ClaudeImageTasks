# From a Small Picture to a Print-Ready Cover

*A worked example in Python, written for anyone who wants to build something
similar themselves.*

---

## Where we're starting

Here's the situation that led to this script. You've made some cover art for a
book. Perhaps you drew it, perhaps you generated it, and either way it looks great
on screen. You go to upload it to Amazon KDP for a printed paperback, and KDP
rejects it, because the image is too small.

The art is 1024 pixels square. KDP wants 300 dots per inch. For an 8.5-inch-square
book, after a few adjustments we'll come to shortly, that means 2588 by 2625
pixels. We need to make the picture about two and a half times bigger *and* keep
it sharp.

That second part is the real problem. Making an image bigger is easy. Making it
bigger without it turning soft and blurry is not.

I'm going to walk you through how I approached this, in roughly the order I'd
want you to think about it. The finished script is `kdp_cover_upscale.py` in this
repository, but please don't start by reading all 400-odd lines of it. Build up
to it with me.

---

## Step 1: Try the obvious thing first

Whenever you meet a new problem, try the simplest possible solution and look
closely at how it fails. The failure tells you what the real problem is.

The simplest solution here is to resize the image with Pillow:

```python
from PIL import Image

img = Image.open("cover.png").convert("RGB")
big = img.resize((2588, 2625), Image.LANCZOS)
big.save("cover_big.png", dpi=(300, 300))
```

Run that, open the result, and zoom in on the title lettering. It's the right
size, but the edges are fuzzy. Fine lines have turned into soft grey smudges.

Why? A resize can only *interpolate*. Each new pixel is a weighted average of the
original pixels near it. There's no new information, just the old information
spread more thinly. Lanczos is about the best interpolation filter you can get,
and it still can't add detail that was never there.

Keep that result. It's the baseline everything else gets compared against.

> **Try this:** resize the same image with `Image.NEAREST`, `Image.BILINEAR` and
> `Image.LANCZOS`, and put crops of each side by side. You'll see the trade-offs
> between blocky, soft and slightly-less-soft for yourself.

---

## Step 2: Understand what "correct" means before improving anything

Before reaching for anything clever, I made sure I knew exactly what the output
needed to be. It's tempting to skip this, and it's the step that saves the most time.

KDP's print rules, in plain terms:

- **300 DPI.** The pixel count must be the inches multiplied by 300.
- **Bleed.** Printers can't cut paper perfectly, so the artwork has to extend
  0.125 inches past the edge where the page is trimmed. That stops a thin white
  sliver appearing if the cut lands a fraction off.
- **The spine edge doesn't need bleed** on a front cover, because that edge folds
  around the spine rather than being cut.
- **A full paperback cover is one wide image:** back cover, spine and front cover
  side by side. The spine's width depends on how many pages the book has and how
  thick the paper is.

So for an 8.5 × 8.5 inch book's front cover:

```
width  = (8.5 + 0.125)     × 300 = 2587.5  →  2588 px   (one bleed edge)
height = (8.5 + 0.125 × 2) × 300 = 2625    →  2625 px   (top and bottom bleed)
```

Notice that I rounded 2587.5 *up*. If we're half a pixel short, we're below the
minimum, so we always round up. In the script, that's this small function:

```python
def px(inches: float, dpi: int) -> int:
    return math.ceil(inches * dpi - 1e-9)
```

The `- 1e-9` looks odd, so it's worth a word. Floating-point maths sometimes gives
`2550.0000000001` when the true answer is exactly 2550, and `ceil` would then bump
it to 2551. Subtracting a tiny amount guards against that. It's a small thing, but
it's the kind of small thing you only learn about by getting bitten by it.

In the script, all of this lives in `build_spec()`. It takes the user's choices
(trim size, page count, paper type) and returns the exact pixel dimensions
*before any image is touched*. I'd encourage you to keep that separation in your
own work: get the arithmetic right and testable on its own, then worry about pixels.

---

## Step 3: Find a tool that adds real detail

This is the heart of it. If interpolation can't add detail, what can?

The answer is **super-resolution**: neural networks trained on millions of pairs
of low- and high-resolution images. They learn what a sharp edge, a line of text
or a scaly texture *usually* looks like at higher resolution, and they predict that
detail. One of the best-known free models is **Real-ESRGAN**, which enlarges by 4×.

You don't need to understand how the network works inside to use it, any more than
you need to understand Lanczos's maths to call `resize`. You do need to know how
to feed it an image and get one back. With the `spandrel` library, which reads
these model files and builds the right network for you, the core looks like this:

```python
import numpy as np
import torch
from spandrel import ModelLoader

model = ModelLoader().load_from_file("RealESRGAN_x4plus.pth").model.eval()

arr = np.asarray(img, dtype=np.float32) / 255.0               # pixels as 0..1
t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)       # (1, 3, H, W)
with torch.inference_mode():
    out = model(t).clamp(0, 1)
out = out[0].permute(1, 2, 0).numpy()                          # back to (H, W, 3)
big = Image.fromarray((out * 255 + 0.5).astype(np.uint8))
```

Take a moment over the middle lines, because they confuse almost everyone the first time.

- Pillow and NumPy store an image as height × width × colour, written `(H, W, 3)`.
- PyTorch models expect colour *first*, and they process a *batch* of images at
  once, so they want `(1, 3, H, W)`.
- `permute(2, 0, 1)` reorders the axes, and `unsqueeze(0)` adds the batch dimension
  of 1. On the way out, we undo both.

Once you understand that shape juggling, most image models stop being mysterious.

Run this on a small image, then compare the result with your Step 1 baseline.
On our book covers, the difference was obvious straight away. The lettering
came back with crisp black outlines, and the line art looked inked rather than smudged.

---

## Step 4: Deal with the problem you'll hit next, which is memory

Run that code on a full 1254-pixel cover and, depending on your machine, it'll
either take an age or run out of memory. The network creates a lot of
intermediate data for every pixel, all at once.

The standard fix is **tiling**: cut the image into small squares, upscale each one,
and stitch the results back together. In the script, the tiles are 192 pixels across.

If you try this naively, though, you'll see faint lines in a grid across the
finished image. Neural networks behave a little strangely at the very edge of
their input, where they can't "see" what lies beyond. Every tile has edges, so
every tile has a slightly wrong border.

The fix I used is simple and worth remembering because it turns up in lots of
image work:

1. Give each tile a margin of extra context, 16 pixels on every side, taken from
   its neighbours.
2. Upscale the tile with its margin.
3. **Throw the upscaled margin away** and keep only the centre.

The network sees enough surrounding detail that the part we keep is accurate.
For tiles at the outer edge of the picture, where there's no neighbour, I pad the
whole image with a mirror reflection first (`np.pad(..., mode="reflect")`) so even
those tiles have something sensible beyond their edge.

The `RealESRGAN.upscale` method in the script is this idea and nothing more. Read
it with this explanation beside you and the slicing arithmetic should make sense.

---

## Step 5: Hit the exact size

Real-ESRGAN always enlarges by exactly 4×. We need about 2.5×. What do we do?

We go past the target and come back. Upscale 4× to roughly 4100 pixels, then use
an ordinary Lanczos resize to bring it *down* to 2588 × 2625. Shrinking an image
is the one thing interpolation does really well, and starting from more detail
than we need gives the sharpest result.

The script's `upscale()` function also handles the other cases:

- If the image is already big enough, it just shrinks it.
- If 4× isn't enough, it runs a second pass.
- If the AI libraries aren't installed, it falls back to Lanczos with a little
  sharpening, so the script still does *something* useful on any machine.

That last point is good practice in general. Treat heavy dependencies like PyTorch
as optional extras, and degrade gracefully when they're missing.

---

## Step 6: The unglamorous parts that make it actually work

At this point you have the clever bit working, and it's tempting to stop. Most of
the remaining lines of the script handle the things that go wrong with real
files. I want to persuade you these matter just as much.

**Images arrive in all sorts of states.** `load_rgb()` deals with them up front:

- photos rotated by phone EXIF data
- PNGs with transparency, which turns black if you save it as JPEG
- images in unusual colour profiles

It always hands back plain RGB, so nothing further down has to worry.

**The picture isn't always the right shape.** A square image going onto a tall
eBook cover has to be cropped or padded. `fit_to_aspect()` offers both. Its
padding option fills the gaps with a blurred, darkened copy of the art, which
looks far better than plain bars. I do this *before* upscaling, while the image
is small, so we don't spend minutes enhancing pixels we'll then crop away.

**Save in the format KDP actually wants:**

- **For print:** a PDF, plus a PNG master copy.
- **For eBooks:** a high-quality JPEG with chroma subsampling turned off, so red
  and blue lettering keeps crisp edges.

**Tell the user what happened.** The script ends by printing:

- the sizes, how far the image was enlarged, and which method was used
- reminders about KDP rules a program can't check, like keeping text away from
  the edges and leaving room for the barcode.

For anything going to print, a person should always have the final look. The
report tells them where to look.

---

## Step 7: Check your work properly

Don't trust your eyes on the full image. Everything looks fine when shrunk to fit
a screen.

What I did instead was cut the same small region (the title, a detailed bit of
line art, a busy colourful area) from both the original and the result, scale the
original crop up to match, and place them side by side. That makes the difference,
and any problems, impossible to miss.

That check caught some things worth knowing:

- Thin lines come out slightly bolder than the original.
- Very fine textures can look a little "painted".
- The final sharpening step left a faint pale halo around some of the lettering.

None of these ruled the result out. You can only make that judgement once you've
actually looked.

Testing also turned up a real bug. I named an output file `flight_front_8.5x8.5`,
and the script saved it as `flight_front_8`. Python's `Path.with_suffix()` had
decided that `.5x8` was a file extension and replaced it. It's a perfectly ordinary
filename, and it would never have occurred to me to test it on purpose. Run your
code on realistic inputs, not just tidy ones.

---

## What I'd do differently

No script is finished. If you build your own version, here's where I'd improve on mine:

- **Write tests for the arithmetic.** `build_spec()` and `px()` take numbers and
  return numbers, so they're perfect for a few `pytest` checks. For example: does
  8.5 × 8.5 really give 2588 × 2625?
- **Verify the downloaded model file.** The script fetches the model weights from
  the internet and loads them. Checking the file against a known SHA-256 hash
  first would protect against a corrupted or tampered download.
- **Don't use `assert` to check user input.** Python skips `assert` statements
  when run with the `-O` flag. Raise a proper error instead.
- **Blend the tiles rather than cutting them.** The margin trick works well, but
  fading overlapping tiles into each other would be safer on smooth gradients like skies.
- **Pass functions only what they need.** Several functions receive the whole
  `args` object from `argparse`, which makes them harder to test on their own.

---

## Your turn

If you want to build something similar, here's the order I'd suggest:

1. Do the naive resize. Save it, and keep it as your baseline.
2. Work out your target dimensions on paper, then in a function, then test that function.
3. Get one AI model working on a small image. Don't add anything else yet.
4. Add tiling once you hit memory limits, and then fix the seams.
5. Only then add the handling for awkward files, the different shapes and the report.
6. At every stage, compare crops side by side.

You'll end up with something that looks a lot like this script. More importantly,
you'll understand *why* each piece is there, because you'll have hit the problem
it solves.

Good luck, and zoom in on everything.
