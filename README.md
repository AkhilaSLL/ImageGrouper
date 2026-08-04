# Image Grouper

Point it at a scattered image dump. It works out which pictures **look alike**,
groups them, and copies them into one destination folder renamed
`<group>_<number>.<ext>` — `1_1.jpg`, `1_2.jpg`, `1_3.png`, `2_1.jpg`, …

Windows desktop app, dark UI (title bar included), single-file `.exe`, no
Python needed to run it.

- Similarity is judged from **pixels only** — shape, colour layout, local
  detail and palette. File size, resolution and metadata are never consulted,
  because none of them survive a resize or a re-encode.
- **Nothing is moved or deleted.** Sources are only ever read, so a run can be
  repeated or abandoned with no consequences.
- The similarity slider **re-groups instantly** after an analysis, without
  re-reading a single file — so you can find the right cut for *your* dump
  by dragging it, not by guessing up front.

```
┌──────────────────────────────────┬────────────────────────────────┐
│ New name   Original      Match   │            PREVIEW             │
│ Group 1  ·  5 files        76%   │  ┌──────┐ ┌──────┐ ┌──────┐    │
│   1_1.jpg  IMG_4035.JPG  100.0%  │  │      │ │      │ │      │    │
│   1_2.jpg  beach_v2.jpg   96.0%  │  └──────┘ └──────┘ └──────┘    │
│   1_3.jpg  beach-warm.jpg 90.5%  │   1_1.jpg  1_2.jpg  1_3.jpg    │
│ Group 2  ·  3 files        92%   │  exemplar     96%      90%     │
└──────────────────────────────────┴────────────────────────────────┘
```

---

## Install

### Option A — download the `.exe` (easiest)

Grab `ImageGrouper.exe` from the
[latest release](https://github.com/AkhilaSLL/ImageGrouper/releases/latest).
It is a single self-contained file: put it anywhere and double-click. No
installer, no Python, no admin rights.

> Windows SmartScreen may warn about an unrecognised publisher because the
> `.exe` is unsigned. Choose **More info → Run anyway**.

### Option B — run from source

```powershell
git clone https://github.com/AkhilaSLL/ImageGrouper.git
cd ImageGrouper
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Requires **Python 3.10+**.

---

## Build the `.exe` yourself

A Windows `.exe` must be built on Windows — PyInstaller cannot cross-compile.

**One click:** double-click `build.bat`. It creates `.venv`, installs
dependencies, generates the icon, and builds the app.

**Manually:**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts\make_icon.py                  # only if assets/ is missing
pyinstaller --noconfirm --clean packaging\ImageGrouper.spec
```

The result is a single standalone file: `dist\ImageGrouper.exe`.

---

## How to use it

### 1. Choose the folders to read

Add **as many source folders as you like** with **Add folder…**. Images are
grouped within each folder *and* across all of them. Select entries and press
**Remove** (or <kbd>Delete</kbd>) to drop them.

- **Include subfolders** toggles recursive scanning.
- **Reuse cached features** skips re-reading files that haven't changed since
  the last run — see [Feature cache](#feature-cache).

### 2. Choose where the organised copies go

Set a **destination** folder. It must sit *outside* every source folder — the
app refuses otherwise, because writing the copies into a folder it scans would
mean the next analysis picks its own output back up.

- **A subfolder per group** puts each group in `group_1\`, `group_2\`, … instead
  of writing everything flat. The file names are the same either way.
- **Images that match nothing** decides what happens to pictures with no
  partner: give each its own group number (the default, so nothing is lost),
  sweep them all into one final group, or leave them out of the copy entirely.

### 3. Analyse, then tune

Click **Analyse**. The button becomes **Cancel** while it runs, and the status
bar reports live throughput and an ETA.

Then **drag the Similarity slider**. Re-grouping runs off the features already
in memory, so it is effectively instant and never touches the disk. Higher =
stricter and more groups; lower = looser and fewer, larger groups.

There is no universally correct setting, and the app does not pretend
otherwise — see [Choosing a threshold](#choosing-a-threshold).

The left panel lists every group and the exact **new name** each file will be
given, so you can see the whole result before anything is written. Selecting a
row shows that group **side by side** in the preview. **Double-click** a row or
a thumbnail to open the image; **right-click** for Open, Show in Explorer and
Copy full path.

### 4. Copy

**Copy N files → destination** writes every grouped image into the destination
under its new name. It asks first, warns if existing files would be
overwritten, and reports anything it could not copy.

Your originals are untouched.

---

## How grouping works

Every image is reduced to four independent descriptions, each producing a
distance between 0 and 1:

| Block | What it captures | Why it is there |
|---|---|---|
| **struct** | dHash (64 bits) + DCT hash (256 bits) | shape and edges, independent of colour |
| **layout** | 8×8 CIELAB thumbnail | *where* the colours sit in the frame |
| **detail** | 12×12 map of local luminance spread | *where* the busy parts are |
| **palette** | saturation-weighted hue histogram + tone histogram | what colours the picture is made of, ignoring position |

They combine as a weighted mean of squares, so the total also lands in
`[0, 1]` and `similarity = 100 × (1 − distance)`:

```
d² = 0.26·struct² + 0.30·layout² + 0.26·detail² + 0.18·palette²
```

Colour work is done in **CIELAB**, not RGB, so "how different are these two
colours" roughly matches what an eye would say. The palette histograms are
stored as square roots of probabilities, which makes plain Euclidean distance
between them the **Hellinger distance** — and interpolated into neighbouring
bins, so a hue drifting over a bin edge doesn't move all its mass at once.

None of those weights are guesses. Each block was measured on real photographs
across four cases — the same file re-encoded, the same scene reframed, unrelated
pictures, and near-uniform content — and the weights are what separated them
best. The `detail` block and the 256-bit DCT hash exist *because* of that last
case: with colour alone, two different scanned pages score **97% similar**,
since a page is white nearly everywhere and the sparse text that distinguishes
it barely moves any cell's average. Its local *variance* moves a great deal.

### Clustering

Groups are built by **average-linkage agglomerative clustering**: two clusters
merge only while the *mean* distance across every pair of their members stays
within the threshold.

The obvious alternative — union-find over all near-enough pairs — is single
linkage, and it chains: A joins C whenever any B happens to sit between them.
On a loose threshold that collapses an entire dump into one enormous group.
Averaging makes a merge answer for the whole cluster rather than for its single
most convenient member.

Within a group, files are numbered from the **exemplar** — the member closest
to the group's centre, shown with a highlighted border — so `<group>_1` is
always the most representative image, and the rest follow in descending
similarity to it. Groups themselves are ordered largest first, then tightest.

### Finding near pairs

Every pair is compared, as one BLAS matrix product per block of rows — the full
n×n distance matrix is never built, only one strip of it at a time, so memory
stays flat as the collection grows.

That is a deliberate choice, not a missing optimisation. An earlier version
carried a projection-bounded search: since projecting onto a unit vector can
only *shrink* a distance (`|⟨u,a⟩ − ⟨u,b⟩| ≤ ‖a − b‖`), sorting by one
principal projection turns the search into a sliding window that provably keeps
every true match. It was correct — it returned exactly the same pairs on 2400
real images at three different radii — and it was **slower every single time**,
at 0.1×, 0.6× and 0.5× the speed of just comparing everything.

The reason is that the window doesn't stay narrow: adding images to a dump
makes the projection denser, so the window spans proportionally more of them.
The clever search is O(n²) as well, only with a Python-level loop where the
plain one has a single BLAS call.

### Choosing a threshold

Measured on real photographs, at the default **75%**:

| Case | Typical similarity |
|---|---|
| Same file, resized or re-encoded | 93 – 99% |
| Same scene, reframed or recoloured | 60 – 88% |
| Different scanned pages / UI screenshots | ~64% |
| Unrelated photographs | ~50% |

Those bands overlap, and no single number separates all of them — that is a
real property of the problem, not a tuning failure. A dump of holiday photos
wants a different cut from a dump of screenshots. This is exactly why the
slider re-groups instantly: set it by looking at the result.

Rough guide: **90%+** near-identical copies only · **75%** the same picture,
edited · **65%** the same scene · **below 55%** things start chaining together.

### Feature cache

Reading images is the expensive part. Descriptors are stored in a small SQLite
database keyed on *path + size + modification time*, so a re-run only decodes
files that actually changed. The cache is strictly an optimisation: a stale row
is detected by the size/mtime check rather than trusted, and if the database
cannot be opened the app simply runs without it.

> Size and mtime are used **only** to detect a changed file, and are never a
> similarity signal.

Untick **Reuse cached features** to force a full re-read, or press
**Clear cache** to discard it. The cache and your settings live in
`%LOCALAPPDATA%\ImageGrouper\`.

---

## Project layout

```
├── build.bat                       one-click Windows build
├── run.py                          launch from a source checkout
├── requirements.txt
├── pyproject.toml
├── assets/                         app.ico, app.png
├── packaging/
│   └── ImageGrouper.spec           PyInstaller build definition
├── scripts/
│   └── make_icon.py                regenerates the icon artwork
├── src/imggroup/
│   ├── engine.py                   descriptors, search, clustering, copying
│   ├── cache.py                    SQLite feature cache
│   ├── settings.py                 settings persisted between launches
│   ├── shellops.py                 open / reveal in Explorer
│   ├── app.py                      main window and controller
│   ├── gallery.py                  side-by-side preview panel
│   ├── theme.py                    dark ttk theme
│   └── resources.py                assets, DPI, dark title bar, taskbar id
└── tests/
    └── test_engine.py              headless engine, cache and settings tests
```

## Development

```powershell
pip install -r requirements.txt
pip install pytest ruff
pytest            # 50 headless tests, no GUI required
ruff check src tests scripts run.py --select E,F,W,B,UP,SIM,I
```

The engine has no GUI dependencies, so `imggroup.engine` can be imported,
tested — or scripted — on its own.

---

## Notes

- Supported formats: JPG, PNG, GIF, BMP, TIFF, WEBP, PPM/PGM, TGA.
- Output extensions are **lower-cased**, so a dump holding both `.JPG` and
  `.jpg` comes out consistent.
- EXIF **orientation** is honoured, because it decides how a picture looks. No
  other metadata is read anywhere.
- Unreadable or corrupt files are skipped and counted in the status bar;
  slightly damaged JPEGs are decoded as far as they will go.
- Reading runs on a thread pool off the UI thread, with a progress bar, live
  throughput and ETA, and a working Cancel button.
- Window size, split position, destination, threshold and options are saved on
  exit and restored next launch. Source folders deliberately are not — each
  launch starts with nothing queued up.
- The title bar follows the app's dark theme on Windows 10/11; older builds
  that don't support the DWM attributes keep the default frame.

## License

[MIT](LICENSE)
