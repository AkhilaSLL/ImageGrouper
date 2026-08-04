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
answers "do these *look alike*?", which is a fuzzier question with no correct
answer — so the design goal is a fast, honest slider the user can tune against
their own dump, not a single clever threshold. Don't "improve" it by importing
the duplicate finder's stricter behaviour.

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
pytest                                 # 50 headless tests, no GUI required
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

**`engine.py` has zero GUI dependencies** and is the only module covered by the
test suite. Everything GUI-related (`app.py`, `gallery.py`, `theme.py`) is
exercised only by manual/probe testing. When changing grouping logic, add a
test rather than re-running the app by hand.

### The descriptor (`engine.py`)

Four blocks, each normalised to a distance in `[0, 1]`, combined as a weighted
mean of squares so the total is also in `[0, 1]`:

```
d² = 0.26·struct² + 0.30·layout² + 0.26·detail² + 0.18·palette²
```

- **struct** — dHash (64 bits) + DCT hash keeping **16×16 = 256 bits**, packed
  into `(5,) uint64` for popcounting.
- **layout** — 8×8 CIELAB thumbnail (192 floats). `srgb_to_lab` is hand-rolled
  numpy because Pillow cannot convert RGB→LAB without an ImageCms profile; it
  is verified against the published sRGB/D65 reference values in the tests.
- **detail** — 12×12 map of per-cell luminance standard deviation over a
  192×192 grey copy (144 floats).
- **palette** — saturation-weighted, circularly-interpolated hue histogram (12
  bins) + interpolated saturation/value histograms (4+4), stored as `sqrt(p)`
  so plain L2 between two of them **is** the Hellinger distance.

**The weights and the block list are measured, not chosen** — `probe/` holds
the scripts. The two findings that shaped the design, and that you should not
undo without re-measuring:

1. The structure hashes are near-**useless** for photographs beyond exact
   re-encodes (same-scene 0.667 vs unrelated 0.679 — indistinguishable). They
   are nonetheless essential, because...
2. ...with colour and mean-colour layout alone, two **different scanned pages
   score 97% similar**, and two different UI screenshots 97.6%. A page is white
   almost everywhere and the text that distinguishes it never moves a cell's
   *mean*. That is why `detail` (a variance map) and the enlarged 256-bit DCT
   hash exist. `test_detail_block_separates_pages_that_colour_cannot` encodes
   this invariant directly.

There is deliberately **no global threshold that separates every case** — the
measured margin stays negative for any weighting. That is a property of the
problem, and the response is the instant re-grouping slider, not more tuning.

**Never reintroduce file size, resolution or metadata as a similarity signal** —
the user ruled them out explicitly, and they don't survive a resize or
re-encode. `size`/`mtime` on `ImgInfo` exist only for the cache's staleness
check and for display. EXIF *orientation* is the one exception and is honoured,
because it decides how the picture looks.

### Search and clustering (`engine.py`)

`scaled_features()` pre-multiplies each block so plain squared Euclidean
distance between two rows equals the weighted continuous part of the distance.
The struct term is added separately from packed bits. Consequently
`‖F_a − F_b‖` is a **lower bound** on the true distance, which is what makes
the fast search sound.

- `_all_pairs_within()` — chunked BLAS all-pairs. Exact, and memory-bounded by
  the block size rather than by n. Capped at `MAX_PAIRS` links, trimmed
  *inside* a block as well as between blocks (a small collection is one single
  block, so a boundary-only check never fires).
  `test_expansion_matches_direct_distances` cross-checks the
  `||a||² + ||b||² − 2a·b` expansion against subtracting the vectors outright,
  which is where its precision is worst.

  **Do not re-add a projection-bounded / LSH search here.** One existed, was
  correct, and was measured slower at every radius tried on 2400 real images
  (0.1×, 0.6×, 0.5×). Its sliding window does not stay narrow as n grows —
  a denser projection spans proportionally more images — so it is O(n²) too,
  with a Python loop where this has one BLAS call. `probe/scale.py` has the
  numbers; re-run it before arguing otherwise.
- `_agglomerate()` — average-linkage (UPGMA) over the near-pair graph, heap
  driven. `sums[x][y]` holds the *total* distance over member pairs, so merging
  is addition. Pairs beyond the search radius were never computed and are
  charged at `radius`, a lower bound; `radius >= cutoff` is what keeps that
  sound. `test_agglomeration_matches_textbook_upgma` cross-checks it against a
  naive UPGMA implementation, and `test_average_linkage_refuses_to_chain` pins
  the reason it isn't union-find.

If you touch either, run `pytest -k "expansion or upgma or chain or cap"`.
These are the safety nets for the remaining optimisations; note that
`test_expansion_matches_direct_distances` carries an explicit "did the fixture
actually produce pairs" guard, because an earlier cross-check passed vacuously
by comparing two empty sets. Don't drop that guard.

`_finalise()` orders groups (largest, then tightest) and picks each group's
exemplar as the member nearest the cluster centre. **Ties are broken on a
rounded distance then on path**, not on raw float noise — in a two-image group
both members are mathematically equidistant from the centre, and comparing raw
floats made the output file names depend on rounding.

### Feature cache (`cache.py`)

SQLite, keyed on `normcase(abspath(path))` with `size` + `mtime_ns` as the
staleness check, consumed via the `CacheLike` protocol so the engine never
imports `cache.py`. Features are stored quantised to one byte per dimension
(~380 bytes/image); the quantisation error is ~0.002, two orders of magnitude
below the distances that decide a grouping.

Every method degrades to a silent no-op on any exception. `clear()` runs
`VACUUM` in a *separate* connection outside the delete's transaction — SQLite
cannot `VACUUM` inside one, and doing so rolls the `DELETE` back.

### GUI (`app.py`, `gallery.py`, `theme.py`, `resources.py`, `shellops.py`)

- `App` owns all state; work runs on background threads that post
  `(kind, payload)` tuples onto a `queue.Queue`, polled every `POLL_MS`.
  Progress messages are coalesced to one repaint per poll.
- **`self.infos` is kept alive after an analysis on purpose.** That is what
  makes the similarity slider re-group without re-reading anything — the single
  most important interaction in the app. `_schedule_regroup` debounces it and
  it runs on the `<ButtonRelease-1>` of the scale, not on every drag pixel.
- The destination is validated against the source folders with `is_inside()` in
  both directions: writing copies into a scanned folder would feed this run's
  output into the next analysis.
- `shellops.py` has **no delete path at all**, unlike the sibling projects.
  This app only reads sources and writes copies, so it needs no Recycle Bin
  handling and should not grow any.

### Packaging

`packaging/ImageGrouper.spec` is `SPECPATH`-relative so it resolves `assets/`
and `src/` regardless of cwd, and excludes heavy packages that numpy/Pillow can
otherwise pull in transitively.

## Probing

`probe/` is gitignored scratch. It holds the calibration scripts behind every
number above (`calibrate.py`, `tune.py`, `detail.py`, `detail2.py`), a dump
generator (`make_dump.py`), in-process app drivers (`drive_app.py`,
`drive_full.py`), a scale check (`scale.py`) and `shot.ps1` (PrintWindow
capture — a plain screen-scrape catches whatever is on top instead, and
PrintWindow works even when the window is on another virtual desktop).

Re-run `calibrate.py` then `tune.py`/`detail2.py` before changing any weight or
block; the tables they print are what the current values were chosen from.
