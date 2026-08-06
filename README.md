# Image Grouper

Point it at a scattered image dump. It works out which pictures **belong
together**, groups them, and copies them into one destination folder renamed
`<group>_<number>.<ext>` — `1_1.jpg`, `1_2.jpg`, `1_3.png`, `2_1.jpg`, …

Windows desktop app, dark UI (title bar included), single-file `.exe`, no
Python and no internet needed to run it.

- Similarity is judged from **what is in the picture**, using a bundled
  DINOv2 vision model — not from file size, resolution or metadata, none of
  which survive a resize or a re-encode.
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
It is a single self-contained file — the vision model is inside it — so put it
anywhere and double-click. No installer, no Python, no admin rights, and it
never touches the network.

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
pyinstaller --noconfirm packaging\ImageGrouper.spec
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
bar reports live throughput and an ETA. Expect roughly **25–35 images per
second** — every picture goes through a neural network, which is far slower
than the colour-histogram approach this replaced and is the reason it works.

Then **drag the Similarity slider**. Re-grouping runs off the features already
in memory, so it is effectively instant and never touches the disk. Higher =
stricter and more groups; lower = looser and fewer, larger groups.

The left panel lists every group and the exact **new name** each file will be
given, so you can see the whole result before anything is written. Selecting a
row shows that group in the preview as a contact sheet, which fills in
progressively — an album of fifty photographs would otherwise lock the window
up while it decoded them all. **Double-click** a row or a thumbnail to open the
image; **right-click** for Open, Show in Explorer and Copy full path.

Each thumbnail is captioned with its new name, its current file name **and the
folder it came from**. An album routinely spans every folder you scanned, so the
folder is what says where a picture actually came from. Very long paths are
shortened from the middle, keeping the drive and the deepest folders — the parts
that actually differ.

### 4. Copy

**Copy N files → destination** writes every grouped image into the destination
under its new name. It asks first, warns if existing files would be
overwritten, and reports anything it could not copy.

Your originals are untouched.

---

## How grouping works

Every image is passed through **DINOv2-small**, a self-supervised vision
transformer, and reduced to a single unit-length 384-number vector. Similarity
is the cosine between two such vectors, rescaled onto a usable 0–100% range:

```
similarity = clip((cos − 0.30) / (1 − 0.30), 0, 1) × 100
```

The point of an embedding rather than hand-crafted statistics is that it
describes *what is in* a picture. Colour layout, perceptual hashes and palette
histograms describe how an image looks **as a grid of numbers**, and that
cannot distinguish "two photographs of the same place" from "two photographs
with the same average colour". That is not a tuning problem — it is what those
features are.

### Why this model, and this recipe

Every choice below was measured on a labelled benchmark of "albums", scored by
mean average precision and by how far apart it pushed same-album and unrelated
pairs. The scripts are in `probe/`.

| Choice | Alternative | Result |
|---|---|---|
| **DINOv2-small** | CLIP ViT-B/32 | mAP **0.653** vs 0.549; separation **0.483** vs 0.120 — and 4× smaller |
| **DINOv2-small** | the old hand-crafted descriptor | mAP 0.653 vs **0.472**; separation 0.483 vs **0.115** |
| **uint8 weights** | fp32 / fp16 | equal within noise (mAP 0.654 vs 0.648) at a quarter of the size |
| **uint8 weights** | q4 | q4 is genuinely worse (0.630) — 7 MB not worth it |
| **Pad to square** | centre-crop | keeps the whole frame; mAP 0.654 vs 0.645, and faster |
| **Raw cosine** | mean-centred cosine | centring's apparent +0.011 AUC vanished on held-out albums |

That last row is worth spelling out, because it is the trap this project fell
into once already. Subtracting the collection mean looked like a clear win —
until it was measured on albums the mean had *not* been computed from, where
the gain disappeared entirely. It also has a fatal edge case: with two images
the mean lies exactly between them, so the centred vectors are exact opposites
and every pair scores 0%.

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
stays flat as the collection grows. Because the embeddings are unit length,
`‖a − b‖² = 2 − 2·cos`, so that single matrix product gives the exact metric
rather than an approximation of it.

That is a deliberate choice, not a missing optimisation. An earlier version
carried a projection-bounded search: since projecting onto a unit vector can
only *shrink* a distance, sorting by one principal projection turns the search
into a sliding window that provably keeps every true match. It was correct —
identical pairs on 2400 real images at three radii — and **slower every single
time**, at 0.1×, 0.6× and 0.5× the speed of just comparing everything, because
the window widens as the projection densifies.

### Choosing a threshold

Measured on the benchmark, by running the real clustering at each setting and
scoring the groups against the true albums:

| Setting | Groups found | Precision | Recall | F1 |
|---|---|---|---|---|
| 85% | 48 | 0.487 | 0.294 | 0.367 |
| 75% | 42 | 0.410 | 0.523 | 0.459 |
| **70%** (default) | **36** | **0.445** | **0.670** | **0.535** |
| 60% | 28 | 0.311 | 0.785 | 0.446 |
| 50% | 21 | 0.207 | 0.887 | 0.335 |

Rough guide: **90%+** near-identical copies only · **80%** the same picture,
edited · **70%** the same scene or subject · **55%** the same broad kind of
thing · **below 50%** things start running together.

There is still no single number that is right for every dump — a folder of
holiday photos wants a different cut from a folder of screenshots. That is why
the slider re-groups instantly: set it by looking at the result.

### Feature cache

Reading and embedding images is by far the expensive part. Vectors are stored
in a small SQLite database keyed on *path + size + modification time*, so a
re-run only re-embeds files that actually changed. The cache is strictly an
optimisation: a stale row is detected by the size/mtime check rather than
trusted, and if the database cannot be opened the app simply runs without it.

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
├── assets/
│   ├── app.ico, app.png
│   └── dinov2-small-uint8.onnx     the bundled vision model (22 MB)
├── packaging/
│   └── ImageGrouper.spec           PyInstaller build definition
├── scripts/
│   └── make_icon.py                regenerates the icon artwork
├── src/imggroup/
│   ├── embedder.py                 preprocessing + the ONNX session
│   ├── engine.py                   scanning, distances, clustering, copying
│   ├── cache.py                    SQLite feature cache
│   ├── settings.py                 settings persisted between launches
│   ├── shellops.py                 open / reveal in Explorer
│   ├── app.py                      main window and controller
│   ├── gallery.py                  contact-sheet preview panel
│   ├── theme.py                    dark ttk theme
│   └── resources.py                assets, DPI, dark title bar, taskbar id
└── tests/
    └── test_engine.py              headless engine, cache and settings tests
```

## Development

```powershell
pip install -r requirements.txt
pip install pytest ruff
pytest            # 48 headless tests, no GUI required
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
  slightly damaged JPEGs are decoded as far as they will go. One bad file never
  takes down the batch it was read in.
- Decoding runs on a thread pool one batch ahead of the model, so reading and
  inference overlap. There is a progress bar, live throughput and ETA, and a
  working Cancel button.
- Window size, split position, destination, threshold and options are saved on
  exit and restored next launch. Source folders deliberately are not — each
  launch starts with nothing queued up.
- The title bar follows the app's dark theme on Windows 10/11; older builds
  that don't support the DWM attributes keep the default frame.

## License

[MIT](LICENSE) — the bundled DINOv2 model is released by Meta under
[Apache 2.0](https://github.com/facebookresearch/dinov2/blob/main/LICENSE).
