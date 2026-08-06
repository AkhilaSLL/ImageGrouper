# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

> Cross-project conventions and machine/environment facts shared with sibling
> projects (e.g. `../DuplicateImageFinder`, `../DuplicateVideoFinder`) live in
> the parent directory's `../CLAUDE.md`, not here. This file only covers what's
> specific to **this** project.

## What this app is *not*

It is not a duplicate finder. `../DuplicateImageFinder` answers "are these the
same file?" and is tuned to say no unless it is nearly certain. This one
answers "do these belong in the same album?", which is a question about picture
*content*, not pixel statistics. Don't "improve" it by importing the duplicate
finder's descriptor or its stricter behaviour — that is exactly what v1 did,
and it is why v1 did not work.

## The v1 failure, and why it matters here

v1 scored similarity from perceptual hashes + CIELAB colour layout + a local
detail map + palette histograms. On a real dump the user reported that at 75%
"nearly 90% were on their own", and that 60–65% both over-grouped and
under-grouped. Both symptoms had one arithmetic cause:

The `struct` block held 26% of the squared distance but, measured over 210 real
pairs, sat between **0.64 and 0.75 for every pair of distinct photographs** —
near-constant. A constant inside a square root is a floor on distance and so a
ceiling on similarity:

```
d >= sqrt(0.26 * 0.687^2) = 0.350   ->   similarity <= 65%
```

Nothing but a byte-identical file could reach the default. The top third of the
slider was unreachable, and at any *usable* setting the struct block was
saturated, so colour alone decided every grouping — hence unrelated pictures
with a shared colour cast grouping while the same scene shot twice did not.

Two lessons that are now encoded as tests, and should not be undone:

1. **A block that barely varies is worse than useless** — it eats budget *and*
   compresses the reachable range. `test_distance_mapping_has_no_constant_floor`
   and `test_a_different_file_can_still_reach_the_top_of_the_slider` pin this.
2. **The calibration data said so at the time.** The v1 notes recorded
   "same-scene 0.667 vs unrelated 0.679 — indistinguishable" and the block was
   shipped at weight 0.26 anyway. If a measurement contradicts a design, the
   measurement wins.

## Commands

```powershell
# Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install pytest ruff pyright        # dev tools, not in requirements.txt

# Run from source
python run.py

# Test
pytest                                 # 48 headless tests, no GUI required
pytest tests/test_engine.py::test_agglomeration_matches_textbook_upgma
pytest -k linkage                      # by keyword

# Lint / type-check
ruff check src tests scripts run.py --select E,F,W,B,UP,SIM,I
pyright                                # config in [tool.pyright] in pyproject.toml

# Build the Windows .exe
python scripts\make_icon.py            # only if assets/ is missing
pyinstaller --noconfirm --clean --distpath dist --workpath build packaging\ImageGrouper.spec
# or just double-click build.bat, which does all of the above from a clean venv
```

`run.py` inserts `src/` onto `sys.path` manually and PyInstaller's `pathex`
does the same, so a new module only needs to live under `src/imggroup/`.

## Architecture

**`engine.py` and `embedder.py` have zero GUI dependencies** and are the only
modules covered by the test suite. Everything GUI-related (`app.py`,
`gallery.py`, `theme.py`) is exercised only by manual/probe testing. When
changing grouping logic, add a test rather than re-running the app by hand.

### The descriptor (`embedder.py`)

One unit-length 384-d vector per image from **DINOv2-small**, uint8 ONNX,
bundled at `assets/dinov2-small-uint8.onnx` (22 MB) and loaded through
onnxruntime's CPU provider. Every choice is measured — `probe/bench.py`,
`probe/variants.py`, `probe/tune_embed.py`, `probe/centring.py` — and the
numbers are in the module docstring and the README. Headlines:

- **DINOv2-small beat CLIP ViT-B/32** on scene albums (mAP 0.653 vs 0.549) and,
  more importantly, separated same-album from unrelated pairs by 0.483 against
  CLIP's 0.120. It is also 4× smaller.
- **uint8 matches fp32 within noise** at a quarter the size; **q4 is worse**
  (mAP 0.630) and not worth the further 7 MB.
- **Pad to square, never centre-crop** — cropping discards 44% of a 16:9 frame.
  Padding is applied *after* normalisation so the border is exactly zero.
- **224px is not tunable.** The exported graph bakes in 257 position
  embeddings; any other input size fails in the position-embedding add.
- **Do not add mean-centring.** It looks like a clear win (AUC 0.9589 → 0.9698)
  but that gain was measured on the same images the mean came from; on held-out
  albums it vanished (0.9818 vs 0.9817). It also makes any two-image collection
  score 0% — the mean sits exactly between them, so the centred vectors are
  exact opposites. `probe/centring.py` demonstrates both.

**Never reintroduce file size, resolution or metadata as a similarity signal** —
the user ruled them out explicitly, and they don't survive a resize or
re-encode. `size`/`mtime` on `ImgInfo` exist only for the cache's staleness
check and for display. EXIF *orientation* is the one exception and is honoured,
because it decides how the picture looks.

### Scale calibration (`engine.py`)

`COSINE_FLOOR = 0.30` rescales cosine onto the slider. It is measured: over the
benchmark, unrelated pairs have median cosine 0.317 and same-album pairs 0.826,
so the floor puts unrelated pairs near 0% and same-album pairs near 75%.
`DEFAULT_THRESHOLD = 70` comes from running the real clustering at every setting
and taking the peak pairwise F1 (0.535, against 0.459 at 75% and 0.446 at 60%) —
see `probe/threshold.py`. Re-run it rather than nudging either constant by feel.

### Search and clustering (`engine.py`)

Embeddings are unit length, so `‖a − b‖² = 2 − 2·cos`: the BLAS search and the
exact cosine metric are the same ordering in different units. `squared_limit()`
converts between them and `test_squared_limit_agrees_with_the_distance_mapping`
pins the conversion — if those two ever disagree the search silently drops real
pairs.

- `_all_pairs_within()` — chunked BLAS all-pairs. Exact, and memory-bounded by
  the block size rather than by n. Capped at `MAX_PAIRS` links, trimmed
  *inside* a block as well as between blocks (a small collection is one single
  block, so a boundary-only check never fires).

  **Do not re-add a projection-bounded / LSH search here.** One existed, was
  correct, and was measured slower at every radius tried on 2400 real images
  (0.1×, 0.6×, 0.5×). Its sliding window does not stay narrow as n grows —
  a denser projection spans proportionally more images — so it is O(n²) too,
  with a Python loop where this has one BLAS call.
- `_agglomerate()` — average-linkage (UPGMA) over the near-pair graph, heap
  driven. `sums[x][y]` holds the *total* distance over member pairs, so merging
  is addition. Pairs beyond the search radius were never computed and are
  charged at `radius`, a lower bound; `radius >= cutoff` is what keeps that
  sound. `test_agglomeration_matches_textbook_upgma` cross-checks it against a
  naive UPGMA implementation, and `test_average_linkage_refuses_to_chain` pins
  the reason it isn't union-find.

If you touch either, run `pytest -k "expansion or upgma or chain or cap"`.
Note that `test_expansion_matches_direct_distances` carries an explicit "did the
fixture actually produce pairs" guard, because an earlier cross-check passed
vacuously by comparing two empty sets. Don't drop that guard.

`_finalise()` orders groups (largest, then tightest) and picks each group's
exemplar as the member nearest the cluster centre, **in float64**. In a
two-image group both members are equidistant from the centre mathematically, but
in float32 the two spreads differ by an ulp — enough to survive rounding and let
noise decide which file becomes `<n>_1`.

### Test fixtures

The fixtures generate pictures of genuinely *different kinds* (text page, bar
chart, sky, noise, colour bands). This is not incidental: the previous fixtures
used one abstract generator with different seeds, which was fine against colour
statistics but is meaningless against an embedding — six pictures of randomly
placed coloured circles scored **81–86% alike**, because to the model they
honestly are the same sort of thing. Same kind/different seed is the
"same album" case; different kind is the "unrelated" case.

`test_unrelated_pictures_score_below_edits_of_one` asserts a *separation*
rather than a magic constant, because individual cross-kind pairs do sometimes
reach the high 70s.

### Feature cache (`cache.py`)

SQLite, keyed on `normcase(abspath(path))` with `size` + `mtime_ns` as the
staleness check, consumed via the `CacheLike` protocol so the engine never
imports `cache.py`. Vectors are stored as float16 (768 bytes/image); the error
moves a cosine by under 1e-5.

`SCHEMA_VERSION` is checked against `PRAGMA user_version` on open and the table
is **dropped** on mismatch. Bump it whenever the stored vector changes meaning:
an old row is not merely stale, it describes the image in a representation that
no longer exists, and reusing it would group by a mixture of two incompatible
feature spaces. `test_rows_from_an_older_schema_are_discarded` covers it.

Every method degrades to a silent no-op on any exception. `clear()` runs
`VACUUM` in a *separate* connection outside the delete's transaction — SQLite
cannot `VACUUM` inside one, and doing so rolls the `DELETE` back.

### GUI (`app.py`, `gallery.py`, `theme.py`, `resources.py`, `shellops.py`)

- `App` owns all state; work runs on background threads that post
  `(kind, payload)` tuples onto a `queue.Queue`, polled every `POLL_MS`.
  Progress messages are coalesced to one repaint per poll.
- **`self.infos` is kept alive after an analysis on purpose.** That is what
  makes the similarity slider re-group without re-reading anything — the single
  most important interaction in the app, and now that embedding costs ~25-35
  img/s it matters far more than it did. `_schedule_regroup` debounces it and
  it runs on the `<ButtonRelease-1>` of the scale, not on every drag pixel.
- **`Gallery._fit_window` must call `update_idletasks()` before measuring.** It
  runs straight after `_render()` grids the cells, before Tk has recomputed the
  frame's requested width — it measured 265px for a group 2656px wide. Worse,
  pinning the canvas window to that stale width stops the frame's *actual* size
  ever changing, so the `<Configure>` binding meant to correct it never fires
  again and most of a group is unreachable at any scroll position. That was a
  real shipped bug; `probe/scrollbug.py` reproduces and verifies it.
- Groups wrap into a **contact sheet** rather than one row once the thumbnails
  would fall below `THUMB_MIN`. Album-sized groups are the normal case now, and
  a single row of eighty images is a 15,000px strip.
- **The preview fills in progressively, and must stay that way.** `_render()`
  lays out the cells (cheap) and then decodes `FILL_CHUNK` thumbnails per idle
  tick. Decoding the whole group before returning — which is what the code did
  when a "group" meant three edits of one photo — blocks the UI thread for as
  long as the group is big: measured 2048ms at sixty small test images, and
  full-size photographs in a real album would clear the ~5s at which Windows
  paints "Not Responding" over the window. Chunked, the worst uninterrupted
  block **plateaus at ~290ms** at every group size, with no change to total
  time. `probe/renderperf.py` measures the right thing — the largest gap
  between heartbeats on the event loop, not the total time to finish drawing.
  Every slot is a fixed box, so late-arriving pictures never move anything
  already on screen. `_reset()` bumps `_generation` *and* cancels `_fill_job`,
  so a queued tick that loses the race still no-ops instead of writing into
  destroyed widgets.
- The destination is validated against the source folders with `is_inside()` in
  both directions: writing copies into a scanned folder would feed this run's
  output into the next analysis.
- The caption carries the file's **containing folder** under its current name
  (`folder_caption`/`elide_middle`). A group here is an album, and an album
  routinely spans every source folder that was scanned, so the folder is what
  says where a picture actually came from. The character budget scales with
  cell width — that matters more here than in the sibling projects, because a
  wrapped contact sheet gives each cell only `THUMB_WRAP` of width to play
  with. `CAPTION_SPACE` was raised to reserve the room.
- `shellops.py` has **no delete path at all**, unlike the sibling projects.
  This app only reads sources and writes copies, so it needs no Recycle Bin
  handling and should not grow any.

### Packaging

`packaging/ImageGrouper.spec` is `SPECPATH`-relative so it resolves `assets/`
and `src/` regardless of cwd. `assets/` carries the ONNX model, so the exe stays
self-contained with nothing to download. **UPX must not touch onnxruntime's
native libraries** — the packed DLL loads but fails at session creation — hence
the `upx_exclude` entries; don't remove them.

## Probing

`probe/` is gitignored scratch. The scripts behind every number above:

| Script | Answers |
|---|---|
| `ceiling.py` | why v1's slider had a hard 65% ceiling |
| `bench.py` | builds the labelled album benchmark; scores any representation |
| `variants.py` | uint8 vs q4 vs fp16 vs fp32: size, speed, accuracy |
| `tune_embed.py` | crop vs squash vs pad, resolution, where the time goes |
| `centring.py` | whether centring generalises (it does not) |
| `calibrate2.py` | the cosine distribution behind `COSINE_FLOOR` |
| `threshold.py` | the clustering F1 sweep behind `DEFAULT_THRESHOLD` |
| `scrollbug.py` | reproduces and verifies the gallery scroll fix |
| `renderperf.py` | worst UI-thread block while the preview renders |
| `drive_v2.py` | drives the real app in process, end to end |
| `shot_app.py` | brings the app up grouped, for a PrintWindow capture |
| `shot_paths.py` | same, but across two space-containing source folders, for the preview's folder line and the Explorer reveal |
| `scale.py` | search timing on a large synthetic dump |

**A GUI probe must never pump Tk by hand around a blocking call.** An earlier
`shot_v2.py` drove the window with `root.update()` in a loop and then made a
synchronous `subprocess.run` to take its own screenshot — with no message pump
running, Windows marked the app "Not Responding" and it had to be killed. Drive
from `after()` callbacks under a real `mainloop()` (as `shot_app.py` does), give
the window an `AUTO_CLOSE_MS` self-destruct so a failed capture cannot strand it
on the desktop, and take the screenshot from a *separate* process.

Re-run `bench.py` before changing the model, the pooling or the preprocessing;
re-run `threshold.py` before changing `COSINE_FLOOR` or `DEFAULT_THRESHOLD`.
