"""
Grouping engine - pure logic, no GUI, independently testable.

Every image becomes one unit-length 384-dimensional vector from DINOv2-small
(see :mod:`imggroup.embedder`, which records why that model and that recipe).
Similarity between two images is the cosine between their vectors, rescaled so
the slider spans a range that is actually reachable::

    similarity = clip((cos - COSINE_FLOOR) / (1 - COSINE_FLOOR), 0, 1)
    distance   = 1 - similarity

**The rescale is the whole point of COSINE_FLOOR, and it is the bug that broke
the previous version.**  That one weighted a perceptual-hash block at 0.26 of
the squared distance - but measured over real photographs the block sat between
0.64 and 0.75 for every pair of *distinct* pictures, near-constant.  A constant
term inside a square root is a floor on the distance, so::

    d >= sqrt(0.26 * 0.687^2) = 0.350   ->   similarity <= 65%

which meant nothing but a byte-identical duplicate could ever score above 65%,
the top third of the slider was unreachable, and at any usable setting the only
block still varying was colour - so unrelated pictures that shared a colour cast
grouped while the same scene shot twice did not.  Measured over 210 real pairs,
exactly 2 reached 75%: the two identical files.

The floor here is measured rather than assumed: over the benchmark in
``probe/bench.py`` unrelated pairs have median cosine 0.317 and same-album pairs
median 0.826, so 0.30 puts unrelated pairs at roughly 0% and same-album pairs
around 75%.  ``test_similarity_scale_is_reachable`` pins it so the failure
cannot come back silently.

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

from .embedder import BATCH_SIZE, EMBED_DIM, Embedder, preprocess, shared

# Slightly damaged JPEGs are common in real image dumps; decode what we can
# instead of discarding the file.
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTS = frozenset({
    ".jpg", ".jpeg", ".jpe", ".png", ".gif", ".bmp",
    ".tif", ".tiff", ".webp", ".ppm", ".pgm", ".tga",
})

# Cosine below this counts as "no more alike than two unrelated pictures".
# Measured, not chosen - see the module docstring and probe/calibrate2.py.
COSINE_FLOOR = 0.30

FEATURE_BYTES = EMBED_DIM * 2            # float16 per dimension in the cache

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
    features: bytes              # EMBED_DIM float16 values


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
    features: np.ndarray = field(repr=False)      # (EMBED_DIM,) unit float32
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

IMAGE_DRAFT = 448        # two model tiles, so a draft decode never undershoots


def load_for_embedding(path: str) -> np.ndarray:
    """Decode ``path`` and return the model's ``(3, 224, 224)`` input."""
    with Image.open(path) as im:
        # No-op unless the codec supports it (JPEG does, PNG does not).  The
        # model only ever sees 224px, so decoding a 24 MP original in full is
        # pure waste.
        with contextlib.suppress(AttributeError, ValueError, OSError):
            im.draft("RGB", (IMAGE_DRAFT, IMAGE_DRAFT))
        # EXIF orientation is honoured because it decides how the picture
        # *looks*; no other metadata is consulted anywhere in this module.
        oriented = ImageOps.exif_transpose(im) or im
        return preprocess(oriented.convert("RGB"))


def fingerprint(path: str, embedder: Embedder | None = None) -> np.ndarray:
    """The embedding for a single image.  Convenience for tests and scripts."""
    batch = load_for_embedding(path)[None, ...]
    return (embedder or shared()).embed(batch)[0]


def pack_features(features: np.ndarray) -> bytes:
    """
    Embedding -> cache bytes.

    float16 keeps ~3 significant digits over a unit-length vector, which moves
    a cosine by well under 0.001 - two orders of magnitude below the gap that
    decides any grouping - at half the size of float32.
    """
    return np.asarray(features, dtype=np.float16).tobytes()


def unpack_features(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.float16).astype(np.float32)


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
    embedder: Embedder | None = None,
) -> tuple[list[ImgInfo], list[tuple[str, str]]]:
    """
    Describe every path.  Returns ``(infos, errors)``, where ``errors`` holds
    ``(path, message)`` for files that could not be read.

    Decoding runs on a thread pool while the model runs on the calling thread;
    the pool is kept one batch ahead so the two overlap.  Decoding is only
    about a tenth of the cost, so there is little more to win there.

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

    total = len(stats)
    cached: dict[str, CachedFeatures] = {}
    if cache is not None:
        with contextlib.suppress(Exception):   # a cache is only ever a speedup
            cached = cache.load(list(stats))

    infos: list[ImgInfo] = []
    todo: list[str] = []
    for path, st in stats.items():
        hit = cached.get(os.path.normcase(os.path.abspath(path)))
        if (hit is not None and hit.size == st.st_size
                and hit.mtime_ns == st.st_mtime_ns
                and len(hit.features) == FEATURE_BYTES):
            infos.append(ImgInfo(path=path, size=st.st_size, mtime=st.st_mtime,
                                 features=unpack_features(hit.features)))
        else:
            todo.append(path)

    done = len(infos)
    if progress is not None and done:
        progress(done, total)

    if todo:
        model = embedder or shared()
        fresh: list[CachedFeatures] = []
        batches = [todo[i:i + BATCH_SIZE]
                   for i in range(0, len(todo), BATCH_SIZE)]

        with ThreadPoolExecutor(
                max_workers=workers or _default_workers()) as pool:
            def submit(index: int) -> list:
                if index >= len(batches):
                    return []
                return [pool.submit(load_for_embedding, p)
                        for p in batches[index]]

            pending = submit(0)
            for index, batch in enumerate(batches):
                if cancel is not None and cancel.is_set():
                    for future in pending:
                        future.cancel()
                    break
                # Start decoding the next batch before blocking on this one's
                # forward pass, so the two overlap.
                ahead = submit(index + 1)

                ready: list[np.ndarray] = []
                kept: list[str] = []
                for path, future in zip(batch, pending, strict=True):
                    try:
                        ready.append(future.result())
                        kept.append(path)
                    except Exception as exc:   # unreadable / unsupported file
                        errors.append((path, str(exc)))
                pending = ahead

                if ready:
                    try:
                        vectors = model.embed(np.stack(ready))
                    except Exception as exc:
                        errors += [(p, str(exc)) for p in kept]
                        vectors = np.empty((0, EMBED_DIM), dtype=np.float32)
                    for path, vector in zip(kept, vectors, strict=False):
                        st = stats[path]
                        infos.append(ImgInfo(path=path, size=st.st_size,
                                             mtime=st.st_mtime,
                                             features=vector))
                        fresh.append(CachedFeatures(
                            key=os.path.normcase(os.path.abspath(path)),
                            size=st.st_size, mtime_ns=st.st_mtime_ns,
                            features=pack_features(vector)))

                done += len(batch)
                if progress is not None:
                    progress(min(done, total), total)

        if cache is not None and fresh:
            with contextlib.suppress(Exception):
                cache.store(fresh)

    # Batches complete out of order relative to the input; sort so results are
    # reproducible.
    infos.sort(key=lambda i: i.path)
    errors.sort()
    return infos, errors


# --------------------------------------------------------------------------- #
#  Distances
# --------------------------------------------------------------------------- #

def stack_features(infos: Sequence[ImgInfo]) -> np.ndarray:
    """``(n, EMBED_DIM)`` matrix of unit-length embeddings."""
    if not infos:
        return np.empty((0, EMBED_DIM), dtype=np.float32)
    matrix = np.empty((len(infos), EMBED_DIM), dtype=np.float32)
    for row, info in enumerate(infos):
        matrix[row] = info.features
    return matrix


def distance_from_cosine(cosine: np.ndarray | float) -> np.ndarray:
    """Cosine similarity -> distance in ``[0, 1]``."""
    similar = (np.asarray(cosine, dtype=np.float64) - COSINE_FLOOR) \
        / (1.0 - COSINE_FLOOR)
    return 1.0 - np.clip(similar, 0.0, 1.0)


def squared_limit(radius: float) -> float:
    """
    Squared *Euclidean* distance matching a ``radius`` in distance units.

    The embeddings are unit length, so ``||a - b||^2 == 2 - 2 cos``.  That
    makes the cheap BLAS search below exact rather than approximate: it is the
    same ordering, just expressed in the metric one matrix product produces.

    A radius of 1 is special.  Every cosine at or below ``COSINE_FLOOR`` clips
    to distance exactly 1.0, so "within 1.0" means *every* pair, including
    opposite vectors - not merely those down to the floor.  Returning the
    algebraic limit there would silently drop the pairs the caller most
    explicitly asked for.
    """
    if radius >= 1.0:
        return float("inf")
    minimum_cosine = COSINE_FLOOR + (1.0 - radius) * (1.0 - COSINE_FLOOR)
    return max(0.0, 2.0 - 2.0 * minimum_cosine)


def pair_distances(features: np.ndarray, left: np.ndarray,
                   right: np.ndarray) -> np.ndarray:
    """Distance for each ``(left[i], right[i])`` pair."""
    if left.size == 0:
        return np.empty(0, dtype=np.float64)
    cosine = np.einsum("ij,ij->i", features[left], features[right])
    return distance_from_cosine(cosine)


def distance(a: ImgInfo, b: ImgInfo) -> float:
    """Distance between two images, in ``[0, 1]``.  Convenience for tests."""
    return float(pair_distances(stack_features([a, b]),
                                np.array([0]), np.array([1]))[0])


def similarity(a: ImgInfo, b: ImgInfo) -> float:
    """How alike two images are, as a percentage."""
    return 100.0 * (1.0 - distance(a, b))


# --------------------------------------------------------------------------- #
#  Near-pair search
# --------------------------------------------------------------------------- #

def _all_pairs_within(features: np.ndarray, radius: float,
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
    rows through cancellation, which is why the caller clamps.
    ``test_expansion_matches_direct_distances`` pins that equivalence.
    """
    n = features.shape[0]
    squared = np.einsum("ij,ij->i", features, features)
    chunk = max(1, min(n, 4_000_000 // max(1, n)))
    limit = squared_limit(radius)

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
    return _confirm(features, left, right, radius)


def _confirm(features: np.ndarray, left: np.ndarray, right: np.ndarray,
             radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact distances for candidate pairs and drop the misses."""
    if left.size == 0:
        return left, right, np.empty(0, dtype=np.float64)
    dist = pair_distances(features, left, right)
    keep = dist <= radius
    return left[keep], right[keep], dist[keep]


def near_pairs(features: np.ndarray, radius: float,
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
    return _all_pairs_within(features, radius, cancel, progress)


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

    features = stack_features(infos)

    # Both stages report 0..100 of their own, so scale them into one bar that
    # only ever moves forwards: searching is the slow part, merging the tail.
    def staged(start: int, width: int) -> ProgressFn | None:
        if progress is None:
            return None
        return lambda done, total: progress(
            start + int(width * done / max(1, total)), 100)

    left, right, dist = near_pairs(features, radius, cancel, staged(0, 70))
    if cancel is not None and cancel.is_set():
        return []

    clusters = _agglomerate(n, left, right, dist, cutoff, radius, cancel,
                            staged(70, 30))
    if cancel is not None and cancel.is_set():
        return []

    return _finalise(infos, features, clusters, ungrouped)


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
              clusters: list[list[int]],
              ungrouped: str) -> list[list[ImgInfo]]:
    """Order the clusters and images inside them, then number everything."""
    real = [c for c in clusters if len(c) > 1]
    singles = [c[0] for c in clusters if len(c) == 1]

    ordered: list[tuple[list[int], float]] = []
    for cluster in real:
        # float64 on purpose.  In a two-image cluster both members are exactly
        # equidistant from the centre *mathematically*, and in float32 the two
        # spreads differ by about an ulp - enough to survive the rounding below
        # and let float noise, rather than the images, decide which file
        # becomes "<n>_1".
        block = features[cluster].astype(np.float64)
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
            features, np.full(len(rest), exemplar, dtype=np.int64),
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
                features, np.full(rest.size, exemplar, dtype=np.int64), rest)
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
