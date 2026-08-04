"""
Grouping engine - pure logic, no GUI, independently testable.

Every image is reduced to four independent blocks, each of which yields a
distance in ``[0, 1]``:

===========  ==========================================================
``struct``   dHash (64 bits) + DCT hash (256 bits).  Shape and edges.
``layout``   8x8 CIELAB thumbnail.  Where the colours sit in the frame.
``detail``   12x12 map of local luminance spread.  Where the busy parts are.
``palette``  Saturation-weighted hue histogram + tone histogram.  What
             colours the picture is made of, ignoring where they are.
===========  ==========================================================

They combine as a weighted mean of squares, so the overall distance also
lands in ``[0, 1]`` and ``similarity = 100 * (1 - distance)``::

    d^2 = Ws*ds^2 + Wl*dl^2 + Wd*dd^2 + Wp*dp^2

The weights are not guesses - see ``probe/`` in the repository, where each
block was measured on real photographs against re-encodes, reframings of one
scene, unrelated pictures, and (the case that forced ``detail`` and the
256-bit DCT hash to exist) near-uniform content like scanned pages and UI
screenshots, where colour says almost nothing.

Grouping is average-linkage agglomerative clustering.  Single-link chaining -
which is what plain union-find gives you - would merge A and C whenever some B
sits between them, and at a loose threshold that collapses an entire dump into
one group.  Requiring the *average* distance across two clusters to clear the
threshold is what stops that.
"""

from __future__ import annotations

import contextlib
import heapq
import os
import shutil
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import numpy as np
from PIL import Image, ImageFile, ImageOps

# Slightly damaged JPEGs are common in real image dumps; decode what we can
# instead of discarding the file.
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTS = frozenset({
    ".jpg", ".jpeg", ".jpe", ".png", ".gif", ".bmp",
    ".tif", ".tiff", ".webp", ".ppm", ".pgm", ".tga",
})

# ---- descriptor geometry -------------------------------------------------- #
DHASH_SIZE = 8                       # -> 8*8  = 64 bits
DCT_SIZE = 32                        # DCT input resolution
PHASH_KEEP = 16                      # -> 16*16 = 256 bits of low frequencies
STRUCT_BITS = DHASH_SIZE * DHASH_SIZE + PHASH_KEEP * PHASH_KEEP      # 320
STRUCT_WORDS = STRUCT_BITS // 64                                     # 5

LAYOUT_GRID = 8                      # 8*8*3 = 192 floats
DETAIL_GRID = 12                     # 12*12 = 144 floats
DETAIL_CELL = 16                     # pixels per detail cell
HUE_BINS = 12
TONE_BINS = 4                        # separately for saturation and value
HUE_SHARE = 0.6                      # hue vs tone split inside the palette
PALETTE_SIZE = 64                    # working resolution for the histograms

LAYOUT_DIMS = LAYOUT_GRID * LAYOUT_GRID * 3
DETAIL_DIMS = DETAIL_GRID * DETAIL_GRID
PALETTE_DIMS = HUE_BINS + 2 * TONE_BINS
FEATURE_DIMS = LAYOUT_DIMS + DETAIL_DIMS + PALETTE_DIMS

# ---- block weights (shares of the squared distance; must sum to 1) -------- #
W_STRUCT = 0.26
W_LAYOUT = 0.30
W_DETAIL = 0.26
W_PALETTE = 0.18

# libjpeg can decode at 1/2, 1/4 or 1/8 scale for free.  The finest thing the
# descriptor looks at is a 192px detail map, so a 256px draft costs nothing.
_DRAFT_TARGET = 256

# Ceiling on retained near pairs, so a hopelessly loose threshold on a huge
# folder degrades into "fewer links found" rather than into swap death.  Each
# pair costs 24 bytes across the three arrays, so this caps them near 144 MB.
MAX_PAIRS = 6_000_000

ProgressFn = Callable[[int, int], None]

UNGROUPED_OPTIONS = (
    "Give each its own group",
    "Put them all in one group",
    "Leave them out",
)


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #

@dataclass(slots=True, frozen=True)
class CachedFeatures:
    """One row of the on-disk feature cache.  See :mod:`imggroup.cache`."""

    key: str                     # normcase(abspath(path))
    size: int
    mtime_ns: int
    bits: bytes                  # STRUCT_BITS packed big-endian
    features: bytes              # FEATURE_DIMS bytes, each a quantised 0..1


class CacheLike(Protocol):
    """The slice of the feature cache that :func:`scan_images` needs."""

    def load(self, paths: Iterable[str]) -> dict[str, CachedFeatures]: ...

    def store(self, entries: Iterable[CachedFeatures]) -> None: ...


@dataclass(slots=True, eq=False)
class ImgInfo:
    """
    One scanned image.

    ``group`` / ``index`` / ``match`` / ``new_name`` are filled in by
    :func:`group_images`; before that they are placeholders.
    """

    path: str
    size: int
    mtime: float
    bits: np.ndarray = field(repr=False)          # (STRUCT_WORDS,) uint64
    features: np.ndarray = field(repr=False)      # (FEATURE_DIMS,) float32
    group: int = 0                                # 1-based group number
    index: int = 0                                # 1-based position in group
    match: float = 0.0                            # % similar to the exemplar

    @property
    def new_name(self) -> str:
        """The name this file gets in the destination folder."""
        return f"{self.group}_{self.index}{ext_of(self.path)}"

    @property
    def date_str(self) -> str:
        return datetime.fromtimestamp(self.mtime).strftime("%Y-%m-%d %H:%M")

    @property
    def size_str(self) -> str:
        return human_size(self.size)

    @property
    def folder(self) -> str:
        return os.path.dirname(self.path)


def ext_of(path: str) -> str:
    """
    Output extension for ``path``.

    Lower-cased on purpose: a dump routinely holds both ``.JPG`` and ``.jpg``,
    and renaming them consistently is the whole point of the exercise.
    """
    return os.path.splitext(path)[1].lower()


def human_size(n: float) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024.0:
            return f"{f:.0f} {unit}" if unit == "B" else f"{f:.1f} {unit}"
        f /= 1024.0
    return f"{f:.1f} TB"


# --------------------------------------------------------------------------- #
#  Fingerprinting
# --------------------------------------------------------------------------- #

def _dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II basis, so ``M @ x @ M.T`` is a 2-D DCT."""
    k = np.arange(n)
    basis = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    basis[0] *= 1.0 / np.sqrt(2.0)
    return basis * np.sqrt(2.0 / n)


_DCT = _dct_matrix(DCT_SIZE)

# sRGB -> XYZ (D65), and the D65 white point.
_RGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
_D65 = np.array([0.95047, 1.00000, 1.08883])


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """``(..., 3)`` sRGB in 0..255 -> ``(..., 3)`` CIELAB (D65 white point)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    t = (linear @ _RGB_TO_XYZ.T) / _D65
    delta = 6.0 / 29.0
    f = np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4.0 / 29.0)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


def dhash_bits(gray: Image.Image) -> np.ndarray:
    """Horizontal gradient hash: is each pixel brighter than its neighbour?"""
    img = gray.resize((DHASH_SIZE + 1, DHASH_SIZE), Image.Resampling.LANCZOS)
    px = np.asarray(img, dtype=np.int16)
    return (px[:, 1:] > px[:, :-1]).ravel()


def phash_bits(gray: Image.Image) -> np.ndarray:
    """
    DCT hash over the low ``PHASH_KEEP`` x ``PHASH_KEEP`` frequencies.

    The usual 8x8 / 64-bit version describes only the coarsest structure, and
    measurement showed it cannot tell two different scanned pages apart.
    Keeping 16x16 costs 24 more bytes per image and roughly halves that error.
    """
    img = gray.resize((DCT_SIZE, DCT_SIZE), Image.Resampling.LANCZOS)
    coefficients = _DCT @ np.asarray(img, dtype=np.float64) @ _DCT.T
    low = coefficients[:PHASH_KEEP, :PHASH_KEEP].ravel()
    return low > np.median(low)


def layout_vector(rgb: Image.Image) -> np.ndarray:
    """Mean CIELAB colour of each cell of an 8x8 grid, scaled to 0..1."""
    small = rgb.resize((LAYOUT_GRID, LAYOUT_GRID), Image.Resampling.BOX)
    lab = srgb_to_lab(np.asarray(small))
    lab[..., 0] /= 100.0                          # L is 0..100
    lab[..., 1:] = (lab[..., 1:] + 110.0) / 220.0  # a/b are about -110..110
    return np.clip(lab, 0.0, 1.0).ravel()


def detail_vector(gray: Image.Image) -> np.ndarray:
    """
    How much local contrast each cell of the frame carries.

    A mean colour map cannot see the difference between two scanned pages:
    both are white nearly everywhere, and the sparse text that distinguishes
    them barely moves any cell's average.  Its *spread* moves a great deal.
    """
    side = DETAIL_GRID * DETAIL_CELL
    px = np.asarray(gray.resize((side, side), Image.Resampling.BOX),
                    dtype=np.float64)
    cells = px.reshape(DETAIL_GRID, DETAIL_CELL,
                       DETAIL_GRID, DETAIL_CELL).swapaxes(1, 2)
    spread = cells.reshape(DETAIL_GRID, DETAIL_GRID, -1).std(axis=-1)
    return np.clip(spread / 64.0, 0.0, 1.0).ravel()


def _spread_hist(values: np.ndarray, bins: int, weights: np.ndarray,
                 circular: bool) -> np.ndarray:
    """
    Histogram with each sample split between its two neighbouring bins.

    Hard binning is brittle: a hue drifting across a bin edge moves all of its
    mass at once, so two shots of one scene can end up as far apart as two
    unrelated pictures.  Interpolating removes that cliff.
    """
    pos = values * bins - 0.5
    lo = np.floor(pos).astype(np.int64)
    frac = pos - lo
    out = np.zeros(bins)
    for index, weight in ((lo, 1.0 - frac), (lo + 1, frac)):
        index = np.mod(index, bins) if circular else np.clip(index, 0, bins - 1)
        np.add.at(out, index, weight * weights)
    return out


def palette_vector(rgb: Image.Image) -> np.ndarray:
    """
    Which colours the picture is made of, as a Hellinger-embeddable vector.

    Hue is weighted by saturation and brightness so a near-grey picture cannot
    vote loudly for whatever arbitrary hue its noise lands on.  Taking the
    square root of the probabilities means plain Euclidean distance between
    two of these *is* the Hellinger distance between the distributions.
    """
    small = rgb.resize((PALETTE_SIZE, PALETTE_SIZE), Image.Resampling.BOX)
    hsv = np.asarray(small.convert("HSV"), dtype=np.float64) / 255.0
    h, s, v = hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel()

    hue = _spread_hist(h, HUE_BINS, s * v, circular=True)
    hue /= max(1e-9, hue.sum())

    ones = np.ones_like(s)
    tone = np.concatenate([_spread_hist(s, TONE_BINS, ones, circular=False),
                           _spread_hist(v, TONE_BINS, ones, circular=False)])
    tone /= max(1e-9, tone.sum())

    return np.sqrt(np.concatenate([HUE_SHARE * hue, (1 - HUE_SHARE) * tone]))


def pack_bits(bits: np.ndarray) -> np.ndarray:
    """Boolean bit array -> ``(STRUCT_WORDS,)`` uint64 for fast popcounting."""
    return np.packbits(bits).view(np.uint64)


def fingerprint(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(bits, features)`` for the image at ``path``."""
    with Image.open(path) as im:
        # No-op unless the codec supports it (JPEG does, PNG does not).
        with contextlib.suppress(AttributeError, ValueError, OSError):
            im.draft("RGB", (_DRAFT_TARGET, _DRAFT_TARGET))
        # EXIF orientation is honoured because it decides how the picture
        # *looks*; no other metadata is consulted anywhere in this module.
        oriented = ImageOps.exif_transpose(im) or im
        rgb = oriented.convert("RGB")
        gray = rgb.convert("L")

        bits = pack_bits(np.concatenate([dhash_bits(gray), phash_bits(gray)]))
        features = np.concatenate([layout_vector(rgb), detail_vector(gray),
                                   palette_vector(rgb)]).astype(np.float32)
        return bits, features


def quantise(features: np.ndarray) -> bytes:
    """Features -> one byte per dimension, for the cache.

    Every block is already scaled to 0..1, so a byte carries the value to
    about 0.002 - two orders of magnitude finer than the distances that decide
    a grouping.
    """
    return np.clip(np.rint(np.asarray(features) * 255.0),
                   0, 255).astype(np.uint8).tobytes()


def dequantise(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 255.0


# --------------------------------------------------------------------------- #
#  Discovery
# --------------------------------------------------------------------------- #

def collect_files(folders: Iterable[str], recursive: bool) -> Iterator[str]:
    """Yield image paths under ``folders``, de-duplicated across folders."""
    seen: set[str] = set()
    for folder in folders:
        if not folder:
            continue
        folder = os.path.abspath(folder)
        if recursive:
            walker = os.walk(folder, onerror=None)
        else:
            try:
                names = os.listdir(folder)
            except OSError:
                continue
            walker = iter([(folder, [], names)])

        for root, _dirs, names in walker:
            for name in names:
                if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
                    continue
                full = os.path.join(root, name)
                key = os.path.normcase(os.path.realpath(full))
                if key in seen:
                    continue
                if not recursive and not os.path.isfile(full):
                    continue
                seen.add(key)
                yield full


# --------------------------------------------------------------------------- #
#  Scanning
# --------------------------------------------------------------------------- #

def _default_workers() -> int:
    # Decoding releases the GIL for most codecs, so oversubscribing a little
    # keeps the CPU busy while other threads wait on disk.
    return max(2, min(16, (os.cpu_count() or 4) + 2))


def scan_images(
    paths: Iterable[str],
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
    workers: int | None = None,
    cache: CacheLike | None = None,
) -> tuple[list[ImgInfo], list[tuple[str, str]]]:
    """
    Describe every path.  Returns ``(infos, errors)``, where ``errors`` holds
    ``(path, message)`` for files that could not be read.

    ``size`` and ``mtime`` are recorded for the cache's staleness check and for
    display only - neither ever reaches :func:`group_images`.
    """
    paths = list(paths)
    errors: list[tuple[str, str]] = []

    stats: dict[str, os.stat_result] = {}
    for path in paths:
        try:
            stats[path] = os.stat(path)
        except OSError as exc:
            errors.append((path, str(exc)))

    todo = list(stats)
    total = len(todo)

    cached: dict[str, CachedFeatures] = {}
    if cache is not None:
        with contextlib.suppress(Exception):   # a cache is only ever a speedup
            cached = cache.load(todo)

    fresh: dict[str, CachedFeatures] = {}
    fresh_lock = threading.Lock()

    def work(path: str) -> ImgInfo:
        st = stats[path]
        hit = cached.get(os.path.normcase(os.path.abspath(path)))
        if (hit is not None and hit.size == st.st_size
                and hit.mtime_ns == st.st_mtime_ns
                and len(hit.features) == FEATURE_DIMS):
            return ImgInfo(path=path, size=st.st_size, mtime=st.st_mtime,
                           bits=np.frombuffer(hit.bits, dtype=np.uint64),
                           features=dequantise(hit.features))

        bits, features = fingerprint(path)
        if cache is not None:
            entry = CachedFeatures(
                key=os.path.normcase(os.path.abspath(path)), size=st.st_size,
                mtime_ns=st.st_mtime_ns, bits=bits.tobytes(),
                features=quantise(features))
            with fresh_lock:
                fresh[entry.key] = entry
        return ImgInfo(path=path, size=st.st_size, mtime=st.st_mtime,
                       bits=bits, features=features)

    infos: list[ImgInfo] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers or _default_workers()) as pool:
        futures = {pool.submit(work, p): p for p in todo}
        try:
            for future in futures:
                if cancel is not None and cancel.is_set():
                    break
                path = futures[future]
                try:
                    infos.append(future.result())
                except Exception as exc:       # unreadable / unsupported file
                    errors.append((path, str(exc)))
                done += 1
                if progress is not None:
                    progress(done, total)
        finally:
            if cancel is not None and cancel.is_set():
                for future in futures:
                    future.cancel()

    if cache is not None and fresh:
        with contextlib.suppress(Exception):
            cache.store(fresh.values())

    # Futures complete out of order; sort so results are reproducible.
    infos.sort(key=lambda i: i.path)
    errors.sort()
    return infos, errors


# --------------------------------------------------------------------------- #
#  Distances
# --------------------------------------------------------------------------- #

_BYTE_POPCOUNT = np.array([bin(i).count("1") for i in range(256)],
                          dtype=np.uint8)


def _popcount(x: np.ndarray) -> np.ndarray:
    """Per-element set-bit count for a uint64 array."""
    counter = getattr(np, "bitwise_count", None)
    if counter is not None:                       # numpy >= 2.0
        return counter(x)
    return _BYTE_POPCOUNT[
        x.view(np.uint8).reshape(x.shape + (8,))].sum(axis=-1)


def scaled_features(infos: Sequence[ImgInfo]) -> np.ndarray:
    """
    Stack the continuous blocks, pre-scaled so plain squared Euclidean
    distance between two rows equals their weighted continuous distance::

        ||F_a - F_b||^2 == Wl*dl^2 + Wd*dd^2 + Wp*dp^2

    Each block's own distance is a root-mean-square over its dimensions (a
    Hellinger distance for the palette), which is why the scale factor carries
    the block size.
    """
    matrix = np.empty((len(infos), FEATURE_DIMS), dtype=np.float32)
    for row, info in enumerate(infos):
        matrix[row] = info.features
    scale = np.empty(FEATURE_DIMS, dtype=np.float32)
    layout_end = LAYOUT_DIMS
    detail_end = layout_end + DETAIL_DIMS
    scale[:layout_end] = np.sqrt(W_LAYOUT / LAYOUT_DIMS)
    scale[layout_end:detail_end] = np.sqrt(W_DETAIL / DETAIL_DIMS)
    scale[detail_end:] = np.sqrt(W_PALETTE / 2.0)
    return matrix * scale


def _struct_distance_sq(bits: np.ndarray, left: np.ndarray,
                        right: np.ndarray) -> np.ndarray:
    """``Ws * ds^2`` for the given index pairs."""
    # Explicit accumulator: popcount yields uint8, and 5 words can total 320.
    hamming = _popcount(bits[left] ^ bits[right]).sum(axis=1, dtype=np.int64)
    return W_STRUCT * hamming.astype(np.float64) / STRUCT_BITS


def pair_distances(features: np.ndarray, bits: np.ndarray, left: np.ndarray,
                   right: np.ndarray) -> np.ndarray:
    """Full distance for each ``(left[i], right[i])`` pair."""
    diff = features[left] - features[right]
    total = np.einsum("ij,ij->i", diff, diff).astype(np.float64)
    total += _struct_distance_sq(bits, left, right)
    return np.sqrt(np.maximum(total, 0.0))


def distance(a: ImgInfo, b: ImgInfo) -> float:
    """Distance between two images, in ``[0, 1]``.  Convenience for tests."""
    features = scaled_features([a, b])
    bits = np.stack([a.bits, b.bits])
    return float(pair_distances(features, bits,
                                np.array([0]), np.array([1]))[0])


def similarity(a: ImgInfo, b: ImgInfo) -> float:
    """How alike two images are, as a percentage."""
    return 100.0 * (1.0 - distance(a, b))


# --------------------------------------------------------------------------- #
#  Near-pair search
# --------------------------------------------------------------------------- #

def _all_pairs_within(features: np.ndarray, bits: np.ndarray, radius: float,
                      cancel: threading.Event | None = None,
                      progress: ProgressFn | None = None,
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Every pair closer than ``radius``.

    Comparison work is one BLAS matrix product per block of rows, and memory is
    bounded by the block size rather than by ``n`` - the full n-by-n distance
    matrix is never materialised, only one strip of it at a time.

    Expanding ``||a - b||^2`` as ``||a||^2 + ||b||^2 - 2 a.b`` is what lets a
    single ``@`` do the work; it can go slightly negative for near-identical
    rows through cancellation, which is why the caller takes a clamped square
    root.  ``test_expansion_matches_direct_distances`` pins that equivalence.
    """
    n = features.shape[0]
    squared = np.einsum("ij,ij->i", features, features)
    chunk = max(1, min(n, 4_000_000 // max(1, n)))
    limit = radius * radius

    lefts: list[np.ndarray] = []
    rights: list[np.ndarray] = []
    found = 0
    for start in range(0, n, chunk):
        if cancel is not None and cancel.is_set():
            break
        stop = min(n, start + chunk)
        block = features[start:stop]
        dist2 = (squared[start:stop, None] + squared[None, :]
                 - 2.0 * (block @ features.T))
        rows = np.arange(start, stop)[:, None]
        hit = (dist2 <= limit) & (np.arange(n)[None, :] > rows)
        ri, ci = np.nonzero(hit)
        # Trim inside the block, not just between blocks: a small collection is
        # one single block, so a boundary-only check would never fire at all.
        capped = found + int(ri.size) >= MAX_PAIRS
        if capped:
            room = max(0, MAX_PAIRS - found)
            ri, ci = ri[:room], ci[:room]
        lefts.append(ri + start)
        rights.append(ci)
        found += int(ri.size)
        if progress is not None:
            progress(stop, n)
        if capped:
            # The threshold is far looser than this collection can support;
            # stop rather than exhaust memory building links nobody asked for.
            break

    if not lefts:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=np.float64)
    left = np.concatenate(lefts)
    right = np.concatenate(rights)
    return _confirm(features, bits, left, right, radius)


def _confirm(features: np.ndarray, bits: np.ndarray, left: np.ndarray,
             right: np.ndarray, radius: float
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact distances for candidate pairs and drop the misses."""
    if left.size == 0:
        return left, right, np.empty(0, dtype=np.float64)
    dist = pair_distances(features, bits, left, right)
    keep = dist <= radius
    return left[keep], right[keep], dist[keep]


def near_pairs(features: np.ndarray, bits: np.ndarray, radius: float,
               cancel: threading.Event | None = None,
               progress: ProgressFn | None = None,
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Index pairs within ``radius``, as ``(left, right, distance)``.

    This compares every pair, and that is a deliberate choice rather than a
    missing optimisation.  An earlier version carried a projection-bounded
    search: because ``|<u, a> - <u, b>| <= ||a - b||``, sorting by one
    principal projection turns the search into a sliding window that provably
    keeps every true match.  It was correct - it returned exactly these pairs
    on 2400 real images at three radii - and it was **slower every time**
    (0.1x, 0.6x and 0.5x the speed of this function).

    The reason is that its window does not stay narrow.  Adding images to a
    dump makes the projection denser, so the window spans proportionally more
    of them: the clever search is O(n^2) as well, only with a Python-level
    loop instead of one BLAS call.  See ``probe/scale.py`` for the numbers
    before reaching for it again.
    """
    n = features.shape[0]
    if n < 2 or radius <= 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=np.float64)
    return _all_pairs_within(features, bits, radius, cancel, progress)


# --------------------------------------------------------------------------- #
#  Clustering
# --------------------------------------------------------------------------- #

def group_images(
    infos: Sequence[ImgInfo],
    similarity_pct: float,
    ungrouped: str = UNGROUPED_OPTIONS[0],
    cancel: threading.Event | None = None,
    progress: ProgressFn | None = None,
) -> list[list[ImgInfo]]:
    """
    Cluster ``infos`` into groups of visually similar images.

    ``similarity_pct`` is the average similarity two clusters must reach to be
    merged.  Each image's ``group``, ``index`` and ``match`` are filled in, so
    ``info.new_name`` is the file name it will be given.
    """
    n = len(infos)
    if n == 0:
        return []

    cutoff = max(0.0, min(1.0, 1.0 - similarity_pct / 100.0))
    # Search a little wider than the cutoff: pairs beyond the search radius are
    # charged at the radius during clustering, so a margin makes that estimate
    # tighter without changing which single pairs qualify.
    radius = min(1.0, cutoff * 1.5 + 0.02)

    features = scaled_features(infos)
    bits = np.stack([info.bits for info in infos]) if n else np.empty((0, 0))

    # Both stages report 0..100 of their own, so scale them into one bar that
    # only ever moves forwards: searching is the slow part, merging the tail.
    def staged(start: int, width: int) -> ProgressFn | None:
        if progress is None:
            return None
        return lambda done, total: progress(
            start + int(width * done / max(1, total)), 100)

    left, right, dist = near_pairs(features, bits, radius, cancel,
                                   staged(0, 70))
    if cancel is not None and cancel.is_set():
        return []

    clusters = _agglomerate(n, left, right, dist, cutoff, radius, cancel,
                            staged(70, 30))
    if cancel is not None and cancel.is_set():
        return []

    return _finalise(infos, features, bits, clusters, ungrouped)


def _agglomerate(n: int, left: np.ndarray, right: np.ndarray,
                 dist: np.ndarray, cutoff: float, radius: float,
                 cancel: threading.Event | None,
                 progress: ProgressFn | None) -> list[list[int]]:
    """
    Average-linkage agglomeration over the near-pair graph.

    Why average and not single linkage: single linkage (equivalently,
    union-find over the near pairs) merges A with C whenever any B happens to
    sit between them, so at a loose threshold a photo dump chains into one
    enormous group.  Requiring the *mean* of every cross-cluster pair to clear
    the threshold makes a merge answer for the whole cluster rather than for
    its single most convenient member.

    ``sums[x][y]`` holds the total distance over every member pair between
    clusters ``x`` and ``y``, so merging is just addition and the average is
    that total over ``|x| * |y|``.  Pairs beyond ``radius`` were never
    computed and are charged at ``radius``, a lower bound on their true
    distance: with ``radius >= cutoff`` that can only make a merge look worse
    than it is, and two clusters sharing no near pair at all can never fall
    below the cutoff, so they are correctly never considered.
    """
    sums: dict[int, dict[int, float]] = {i: {} for i in range(n)}
    for a, b, d in zip(left.tolist(), right.tolist(), dist.tolist(),
                       strict=True):
        sums[a][b] = d
        sums[b][a] = d

    sizes = dict.fromkeys(range(n), 1)
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    heap = [(float(d), int(a), int(b))
            for a, b, d in zip(left.tolist(), right.tolist(),
                               dist.tolist(), strict=True)
            if d <= cutoff]
    heapq.heapify(heap)

    next_id = n
    merges = 0
    target = max(1, len(heap))
    while heap:
        if cancel is not None and cancel.is_set():
            return []
        mean, a, b = heapq.heappop(heap)
        if mean > cutoff:
            break
        # A live pair's stored sum never changes - only a merge can alter it,
        # and that retires both ids - so "still adjacent" is the whole
        # staleness check.
        if a not in sums or b not in sums or b not in sums[a]:
            continue

        merged = next_id
        next_id += 1
        size = sizes[a] + sizes[b]
        neighbours = (set(sums[a]) | set(sums[b])) - {a, b}

        combined: dict[int, float] = {}
        for other in neighbours:
            total = (sums[a].get(other, sizes[a] * sizes[other] * radius)
                     + sums[b].get(other, sizes[b] * sizes[other] * radius))
            combined[other] = total
            sums[other].pop(a, None)
            sums[other].pop(b, None)
            sums[other][merged] = total
            average = total / (size * sizes[other])
            if average <= cutoff:
                heapq.heappush(heap, (average, merged, other))

        del sums[a], sums[b]
        sums[merged] = combined
        sizes[merged] = size
        members[merged] = members.pop(a) + members.pop(b)

        merges += 1
        if progress is not None and merges % 64 == 0:
            progress(min(merges, target), target)

    return [sorted(members[key]) for key in sums]


def _finalise(infos: Sequence[ImgInfo], features: np.ndarray,
              bits: np.ndarray, clusters: list[list[int]],
              ungrouped: str) -> list[list[ImgInfo]]:
    """Order the clusters and images inside them, then number everything."""
    real = [c for c in clusters if len(c) > 1]
    singles = [c[0] for c in clusters if len(c) == 1]

    ordered: list[list[int]] = []
    for cluster in real:
        block = features[cluster]
        centre = block.mean(axis=0)
        offset = block - centre
        spread = np.einsum("ij,ij->i", offset, offset)
        # The most typical member leads the group, so "<n>_1" is the one worth
        # looking at first.  Rounding before the comparison matters: in a pair
        # both members are equidistant from the centre *mathematically*, and
        # picking on raw float noise would make the output names depend on
        # rounding rather than on the images.
        exemplar = min(
            zip(spread.tolist(), (infos[i].path for i in cluster), cluster,
                strict=True),
            key=lambda item: (round(item[0], 9), item[1]))[2]
        rest = [i for i in cluster if i != exemplar]
        distances = pair_distances(
            features, bits, np.full(len(rest), exemplar, dtype=np.int64),
            np.array(rest, dtype=np.int64)) if rest else np.empty(0)
        paired = sorted(zip(rest, distances.tolist(), strict=True),
                        key=lambda item: (item[1], infos[item[0]].path))
        ordered.append(([exemplar] + [i for i, _d in paired],
                        float(np.mean(distances)) if rest else 0.0))

    # Biggest groups first, then the tightest, then by path for determinism.
    ordered.sort(key=lambda item: (-len(item[0]), item[1],
                                   infos[item[0][0]].path))
    laid_out = [cluster for cluster, _tightness in ordered]

    singles.sort(key=lambda i: infos[i].path)
    if ungrouped == UNGROUPED_OPTIONS[1] and singles:
        laid_out.append(singles)
    elif ungrouped != UNGROUPED_OPTIONS[2]:
        laid_out += [[i] for i in singles]

    groups: list[list[ImgInfo]] = []
    for number, cluster in enumerate(laid_out, start=1):
        exemplar = cluster[0]
        members = [infos[i] for i in cluster]
        rest = np.array(cluster[1:], dtype=np.int64)
        if rest.size:
            distances = pair_distances(
                features, bits, np.full(rest.size, exemplar, dtype=np.int64),
                rest)
        else:
            distances = np.empty(0)
        for position, info in enumerate(members):
            info.group = number
            info.index = position + 1
            info.match = 100.0 if position == 0 else round(
                100.0 * (1.0 - float(distances[position - 1])), 1)
        groups.append(members)
    return groups


# --------------------------------------------------------------------------- #
#  Organising
# --------------------------------------------------------------------------- #

def destination_conflicts(groups: Sequence[Sequence[ImgInfo]],
                          destination: str) -> list[str]:
    """Names in ``destination`` that an organise run would have to overwrite."""
    existing: set[str] = set()
    with contextlib.suppress(OSError):
        existing = {os.path.normcase(n) for n in os.listdir(destination)}
    return [info.new_name for group in groups for info in group
            if os.path.normcase(info.new_name) in existing]


def is_inside(child: str, parent: str) -> bool:
    """Is ``child`` the same folder as ``parent``, or underneath it?"""
    try:
        c = os.path.normcase(os.path.abspath(os.path.realpath(child)))
        p = os.path.normcase(os.path.abspath(os.path.realpath(parent)))
    except OSError:                             # pragma: no cover - unlikely
        return False
    return c == p or c.startswith(p + os.sep)


def organise(
    groups: Sequence[Sequence[ImgInfo]],
    destination: str,
    subfolders: bool = False,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> tuple[int, int, list[tuple[str, str]]]:
    """
    Copy every grouped image into ``destination`` under its new name.

    Returns ``(copied, bytes_copied, errors)``.  Sources are only ever read -
    nothing is moved or deleted - so a run can be repeated or abandoned
    safely.
    """
    total = sum(len(group) for group in groups)
    copied = 0
    written = 0
    errors: list[tuple[str, str]] = []
    done = 0

    for group in groups:
        if cancel is not None and cancel.is_set():
            break
        folder = destination
        if subfolders and group:
            folder = os.path.join(destination, f"group_{group[0].group}")
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError as exc:
            errors += [(info.path, str(exc)) for info in group]
            done += len(group)
            if progress is not None:
                progress(done, total)
            continue

        for info in group:
            if cancel is not None and cancel.is_set():
                break
            target = os.path.join(folder, info.new_name)
            try:
                shutil.copy2(info.path, target)
                copied += 1
                written += info.size
            except Exception as exc:
                errors.append((info.path, str(exc)))
            done += 1
            if progress is not None:
                progress(done, total)

    return copied, written, errors
