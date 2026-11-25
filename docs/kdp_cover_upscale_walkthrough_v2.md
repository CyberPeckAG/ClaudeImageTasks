# `kdp_cover_upscale.py`: annotated walkthrough (v2)

Read this with the script open beside it. Line numbers match `kdp_cover_upscale.py`
as of commit `d3e0d53`.

**The problem:** a 1024 px image needs to become 2588 × 2625 px (8.5" square + bleed
at 300 DPI) *and stay sharp*.

**How to read it:** start at `main()`, follow it down, and jump out to each function as it's called.

---

## 0. Before reading any code: run it

```bash
python kdp_cover_upscale.py cover.png --target print-front --trim 8.5x8.5
```

Zoom in on the output's lettering next to the original. That difference is what the
whole file exists for.

---

## 1. Start at the bottom: `main()`, L369–435

**L438–439**: `sys.exit(main())`. `main` returns an exit code instead of exiting
itself, so a test can call `main(["x.png", "--target", "ebook"])` directly.

**L370–391**: the command-line options. Skim them now; each one comes up again later.
- **L374** `type=parse_trim`: argparse calls *your* function to convert "6x9" into
  `(6.0, 9.0)`. If it raises, argparse prints a clean error. Jump to L82–91 to see it.
- **L373, L376, L385**: `choices=[...]` means argparse rejects typos for free.

**L394–402**: validation argparse can't do on its own.
- ⚠ **L400**: `assert` used to check input. **Don't copy this.** Running Python with
  `-O` strips asserts. Raise an error instead.

**L404–434**: the pipeline. *This is the whole program in 30 lines.*

```
L404  build_spec      what size must the output be?
L405  load_rgb        get a clean RGB image
L414  fit_to_aspect   make it the right shape (still small)
L416  upscale         make it the right size (the hard part)
L417  sharpen         optional
L420  wrap assembly   only if building back + spine + front
L433  save_outputs
L434  report
```

Now take them one at a time.

---

## 2. `build_spec()`, L98–133: do the maths before touching pixels

**L94–95** `px()`. Converts inches to pixels and **rounds up**, because half a pixel
short means below KDP's minimum.
- Why `- 1e-9`? Float noise. Try `0.07 * 300` in Python: you get `21.000000000000004`,
  and `ceil` would give 22. The tiny subtraction stops that.

**L105–112**: front cover.
- Width = trim + **one** bleed: the spine edge folds, so it isn't cut.
- Height = trim + **two** bleeds: top and bottom are both cut.
- Check it yourself: `(8.5 + 0.125) × 300 = 2587.5 → 2588`.

**L118–122**: full wrap = back + spine + front.
- Spine width = pages × paper thickness (the table at **L42–47**).
- **L121**: `2 * side_px + spine_px`. Draw this on paper if it isn't obvious.

**L131–132** `front_box`: if the user only has front art, record where it sits.
**L420–422** uses it later.

> **Why separate?** This function takes numbers and returns numbers, so it can be
> tested without any images. Get the arithmetic right first.

---

## 3. `load_rgb()`, L139–159: clean the input once

Each block fixes one way real files go wrong:

| Lines | Problem | Fix |
| --- | --- | --- |
| L142 | Phone photos stored sideways | `exif_transpose` |
| L144–147 | Transparency turns black in JPEG | Composite onto white |
| L148–156 | Odd colour profile | Convert to sRGB |
| L157–158 | Anything else (greyscale, palette) | Force RGB |

> **The point:** after L159, no other function has to care what format the input was.

---

## 4. `fit_to_aspect()`, L165–191: shape before size

**L169**: already the right shape? Do nothing.

**L171–178** `crop`: cut the excess.
- L172: too wide → trim the sides. Otherwise (L176) trim top and bottom.
- `gravity` (0–1) chooses *which* part survives: 0 = left/top, 0.5 = centre.

**L180–190** `pad`: grow the canvas instead.
- L184–187: enlarge a copy until it covers the canvas, then centre-crop it.
- L188: blur it heavily. L189: darken it to 80%.
- L190: paste the sharp original on top.

> **Why before upscaling?** The image is still small, so this is cheap, and we
> don't spend minutes AI-enhancing pixels we're about to crop off.

---

## 5. `upscale()`, L272–303: the decision-maker

**L274**: `factor` = how much bigger we need. Uses `max` so *both* dimensions reach target.

**L275–276**: already big enough → just shrink.

**L279–283**: `auto` picks the AI if it's installed, otherwise falls back. No crash.
**L212–218** `ai_available()` simply *tries* the import. That's the standard
optional-dependency trick.

**L289–291**: the AI always does 4×. Loop at most twice (16×).

**L293–296** `--strength`: blend the AI result with a plain resize. It's a dial for
"too much invented detail".

**L301–302**: overshoot, then come back. 4× gives ~4100 px, and Lanczos shrinks it to
exactly 2588. Shrinking is what interpolation is good at.

---

## 6. `RealESRGAN`, L221–259: the clever bit

### Setup, L224–236
- **L229**: GPU if available, otherwise CPU.
- **L230**: `spandrel` reads the weights file and builds the right network. You don't
  need to know the architecture.
- **L233–235**: half precision on GPU only. It's faster, and on CPU it isn't supported well.

### `upscale()`, L238–259: slow down here

**L240**: Pillow → NumPy, values 0–255 → 0.0–1.0. Shape is `(H, W, 3)`.

**L244**: mirror-pad the *whole* image by 16 px, so edge tiles have something beyond them.

**L246**: list the top-left corner of every 192 px tile.

**L249**: cut the tile *plus* 16 px of margin on every side.

**L250**: ⭐ the line students get stuck on.
```
torch.from_numpy(patch)   (H, W, 3)
  .permute(2, 0, 1)       (3, H, W)     PyTorch wants colour first
  .unsqueeze(0)           (1, 3, H, W)  plus a batch dimension
```

**L253–254**: run the model. `inference_mode` = no gradient tracking (faster, less memory).
`clamp_(0, 1)` trims overshoot.

**L255**: undo L250. Back to `(H, W, 3)`.

**L256**: ⭐ **throw the margin away**. Only the centre of each upscaled tile is kept.
- Why? Networks guess badly at the edge of their input. With a margin, the part we
  keep was never at an edge, so no seams.
- Everything is `× s` because the output is 4× bigger.

**L259**: back to 0–255 integers. `+ 0.5` makes the cast round instead of truncate.

> **Ask yourself:** what would you see if L256 kept the whole tile, margin included?
> (Answer: a faint grid across the image.)

---

## 7. `lanczos_upscale()`, L262–269: the fallback

At most 2× per step (L265), with a light sharpen after each (L267). Small steps
keep edges cleaner than one big jump. It's still interpolation, so no new detail.

---

## 8. Saving and reporting, L309–366

**L325** `subsampling=0`: keep full colour resolution, so red and blue lettering
stays crisp.
**L324–328**: if the JPEG is over KDP's limit, drop the quality 5 at a time.
**L331–336**: print gets a PDF (for KDP) and a PNG master.

**L309–315** `parse_color`: with no back-cover colour given, average the front's
left strip (the spine side), so the colours match across the fold.

**L340–366** `report`: print what happened, then warn about things code can't check
(the barcode area, safe margins). A human signs off on print.

---

## 9. One bug worth showing: L427–429

The original code was `args.output.with_suffix("")`.

- Input: `-o output/flight_front_8.5x8.5`
- `with_suffix` thought `.5x8` was a file extension
- Saved as `flight_front_8.pdf` ✗

**Fix (L428–429):** only strip suffixes that really are image extensions.

> **Lesson:** this only showed up when testing with a *real* filename. Test with
> realistic inputs, not tidy ones.

---

## 10. What I'd improve

- **L94, L98**: no tests. `px()` and `build_spec()` are the easy first targets.
- **L197–209**: downloaded weights aren't hash-checked. Add a SHA-256 check.
- **L400**: replace the `assert`.
- **L256**: tiles are hard-cut. Feathered blending would be safer on smooth skies.
- **L272, L318, L340**: these take the whole `args` object. Pass only the values they need.

---

## Exercises

1. Change L256 to keep the margin, run it, and find the seams.
2. Write a `pytest` test proving 8.5 × 8.5 gives 2588 × 2625.
3. Run with `--engine lanczos` and with the default AI engine, then compare crops side by side.
4. Try `--strength 0.5`. When would you want that?
