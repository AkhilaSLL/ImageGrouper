"""
Headless tests for the grouping engine, cache and settings.  No GUI is
imported anywhere in here, so the whole suite runs without a display.
"""

from __future__ import annotations

import itertools
import os
import random

import numpy as np
import pytest
from PIL import Image, ImageDraw

from imggroup import engine
from imggroup.cache import FeatureCache
from imggroup.engine import (
    UNGROUPED_OPTIONS,
    CachedFeatures,
    ImgInfo,
    collect_files,
    dequantise,
    destination_conflicts,
    group_images,
    is_inside,
    organise,
    quantise,
    scan_images,
    similarity,
    srgb_to_lab,
)
from imggroup.settings import Settings

# --------------------------------------------------------------------------- #
#  Image fixtures
# --------------------------------------------------------------------------- #

def gradient(path: str, top: tuple[int, int, int],
             bottom: tuple[int, int, int], size: int = 300,
             **save) -> str:
    """A smooth vertical gradient - stands in for "a photo of a sky"."""
    image = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(image)
    for y in range(size):
        t = y / max(1, size - 1)
        draw.line([(0, y), (size, y)],
                  fill=tuple(int(top[i] + (bottom[i] - top[i]) * t)
                             for i in range(3)))
    image.save(path, **save)
    return path


def scene(path: str, seed: int, size: int = 300, **save) -> str:
    """A distinctive picture: coloured background plus placed shapes."""
    rng = random.Random(seed)
    base = (rng.randrange(40, 220), rng.randrange(40, 220),
            rng.randrange(40, 220))
    image = Image.new("RGB", (size, size), base)
    draw = ImageDraw.Draw(image)
    for _shape in range(6):
        x, y = rng.randrange(0, size - 60), rng.randrange(0, size - 60)
        colour = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        draw.ellipse([x, y, x + rng.randrange(30, 90),
                      y + rng.randrange(30, 90)], fill=colour)
    image.save(path, **save)
    return path


def page(path: str, seed: int) -> str:
    """A synthetic scanned page: white, with text blocks in varying places."""
    rng = random.Random(seed)
    image = Image.new("RGB", (420, 560), (252, 251, 248))
    draw = ImageDraw.Draw(image)
    y = rng.randint(40, 90)
    while y < 510:
        for _line in range(rng.randint(2, 6)):
            if y > 515:
                break
            draw.rectangle([50, y, 50 + rng.randint(120, 330), y + 6],
                           fill=(38, 38, 44))
            y += 14
        y += rng.randint(18, 48)
    image.save(path)
    return path


def describe(path: str) -> ImgInfo:
    infos, errors = scan_images([path])
    assert not errors, errors
    return infos[0]


# --------------------------------------------------------------------------- #
#  Colour maths
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("rgb,expected", [
    ((255, 255, 255), (100.00, 0.00, 0.00)),
    ((0, 0, 0), (0.00, 0.00, 0.00)),
    ((255, 0, 0), (53.24, 80.09, 67.20)),
    ((0, 255, 0), (87.73, -86.18, 83.18)),
    ((0, 0, 255), (32.30, 79.19, -107.86)),
    ((128, 128, 128), (53.59, 0.00, 0.00)),
])
def test_srgb_to_lab_matches_the_standard(rgb, expected):
    got = srgb_to_lab(np.array(rgb, dtype=float))
    assert np.allclose(got, np.array(expected), atol=0.01)


def test_lab_conversion_is_vectorised_over_images():
    pixels = np.random.default_rng(0).integers(0, 256, size=(4, 5, 3))
    bulk = srgb_to_lab(pixels)
    assert bulk.shape == (4, 5, 3)
    for y, x in itertools.product(range(4), range(5)):
        assert np.allclose(bulk[y, x], srgb_to_lab(pixels[y, x]))


def test_palette_is_a_unit_probability_vector(tmp_path):
    """sqrt(p) vectors are what make plain L2 a Hellinger distance."""
    with Image.open(scene(str(tmp_path / "p.png"), seed=3)) as image:
        vector = engine.palette_vector(image.convert("RGB"))
    assert vector.shape == (engine.PALETTE_DIMS,)
    assert np.isclose(float(vector @ vector), 1.0, atol=1e-6)


# --------------------------------------------------------------------------- #
#  Descriptor behaviour
# --------------------------------------------------------------------------- #

def test_identical_files_are_100_percent(tmp_path):
    first = scene(str(tmp_path / "a.png"), seed=1)
    second = str(tmp_path / "b.png")
    Image.open(first).save(second)
    assert similarity(describe(first), describe(second)) == pytest.approx(100.0)


def test_reencoding_and_resizing_stays_highly_similar(tmp_path):
    original = scene(str(tmp_path / "orig.png"), seed=2, size=600)
    with Image.open(original) as image:
        rgb = image.convert("RGB")
        rgb.resize((300, 300), Image.Resampling.LANCZOS).save(
            tmp_path / "half.jpg", quality=70)
        rgb.save(tmp_path / "q35.jpg", quality=35)

    base = describe(original)
    for name in ("half.jpg", "q35.jpg"):
        score = similarity(base, describe(str(tmp_path / name)))
        assert score > 90.0, f"{name} scored {score:.1f}%"


def test_unrelated_pictures_are_not_similar(tmp_path):
    a = describe(scene(str(tmp_path / "a.png"), seed=10))
    b = describe(scene(str(tmp_path / "b.png"), seed=99))
    assert similarity(a, b) < 75.0


def test_detail_block_separates_pages_that_colour_cannot(tmp_path):
    """
    The case that forced the detail map and the 256-bit DCT hash to exist:
    two different scanned pages are white nearly everywhere, so palette and
    mean-colour layout both call them near-identical.
    """
    first, second = describe(page(str(tmp_path / "p1.png"), 1)), \
        describe(page(str(tmp_path / "p2.png"), 2))

    colour_only = np.concatenate([
        first.features[:engine.LAYOUT_DIMS],
        first.features[engine.LAYOUT_DIMS + engine.DETAIL_DIMS:]])
    other_colour = np.concatenate([
        second.features[:engine.LAYOUT_DIMS],
        second.features[engine.LAYOUT_DIMS + engine.DETAIL_DIMS:]])
    colour_gap = float(np.abs(colour_only - other_colour).mean())

    detail = first.features[engine.LAYOUT_DIMS:
                            engine.LAYOUT_DIMS + engine.DETAIL_DIMS]
    other_detail = second.features[engine.LAYOUT_DIMS:
                                   engine.LAYOUT_DIMS + engine.DETAIL_DIMS]
    detail_gap = float(np.abs(detail - other_detail).mean())

    assert detail_gap > colour_gap * 3, (
        f"detail {detail_gap:.4f} barely beats colour {colour_gap:.4f}; "
        "the descriptor has lost the only block that sees text placement")


def test_quantisation_round_trip_is_faithful():
    rng = np.random.default_rng(4)
    values = rng.random(engine.FEATURE_DIMS).astype(np.float32)
    back = dequantise(quantise(values))
    assert back.shape == values.shape
    assert np.max(np.abs(back - values)) < 1.0 / 255.0


# --------------------------------------------------------------------------- #
#  Near-pair search
# --------------------------------------------------------------------------- #

def _synthetic_population(n: int, seed: int = 0):
    """
    Clustered feature vectors plus matching structure bits.

    Both blocks have to be tuned together: the structure term contributes
    ``0.26 * hamming / 320`` to the squared distance, so wholly random bits
    would put *every* pair past even the widest radius under test and leave
    the checks below comparing two empty sets.  Cluster members differ in a few
    bits, different clusters in a few dozen.
    """
    rng = np.random.default_rng(seed)
    centres = rng.random((max(2, n // 25), engine.FEATURE_DIMS))
    pick = rng.integers(0, centres.shape[0], size=n)
    features = (centres[pick]
                + rng.normal(0, 0.02, (n, engine.FEATURE_DIMS))).astype(
                    np.float32)
    features *= np.float32(np.sqrt(engine.W_LAYOUT / engine.FEATURE_DIMS))

    base = rng.integers(0, 2 ** 20, size=(centres.shape[0],
                                          engine.STRUCT_WORDS),
                        dtype=np.int64).astype(np.uint64)
    noise = rng.integers(0, 2 ** 3, size=(n, engine.STRUCT_WORDS),
                         dtype=np.int64).astype(np.uint64)
    return features, base[pick] ^ noise


def _as_pairs(left, right, dist):
    return {(int(a), int(b)): round(float(d), 9)
            for a, b, d in zip(left, right, dist, strict=True)}


@pytest.mark.parametrize("radius", [0.10, 0.25, 0.40])
def test_expansion_matches_direct_distances(radius):
    """
    The safety net for the one optimisation left in the search.

    Distances come from expanding ``||a-b||^2`` as ``||a||^2 + ||b||^2 - 2a.b``
    so that a whole block is one BLAS product.  That form loses precision
    exactly where it matters most - near-identical images, where the two large
    terms cancel - so check it against subtracting the vectors outright.
    """
    n = 400
    features, bits = _synthetic_population(n, seed=7)
    got = _as_pairs(*engine._all_pairs_within(features, bits, radius))

    # Straightforward, slow, and numerically unambiguous.
    want = {}
    for i in range(n):
        for j in range(i + 1, n):
            diff = features[i] - features[j]
            hamming = int(np.unpackbits(
                (bits[i] ^ bits[j]).view(np.uint8)).sum())
            distance = float(np.sqrt(
                float(diff @ diff)
                + engine.W_STRUCT * hamming / engine.STRUCT_BITS))
            if distance <= radius:
                want[(i, j)] = round(distance, 9)

    assert set(got) == set(want), (
        f"radius {radius}: {len(set(want) - set(got))} pairs missed, "
        f"{len(set(got) - set(want))} invented")
    # Guard against the check passing because both sides found nothing.
    assert len(want) > 50, f"radius {radius} only produced {len(want)} pairs"
    for key, distance in want.items():
        assert got[key] == pytest.approx(distance, abs=1e-6)


def test_near_pairs_caps_runaway_link_counts(monkeypatch):
    """A threshold loose enough to link everything must not eat all memory."""
    features, bits = _synthetic_population(600, seed=8)
    monkeypatch.setattr(engine, "MAX_PAIRS", 500)
    left, _right, _dist = engine.near_pairs(features, bits, 1.0)
    assert 0 < left.size <= 500
    assert left.size < 600 * 599 // 2, "the cap did not engage at all"


# --------------------------------------------------------------------------- #
#  Clustering:  fast path vs brute force
# --------------------------------------------------------------------------- #

def reference_average_linkage(matrix: np.ndarray,
                              cutoff: float) -> list[list[int]]:
    """
    Textbook UPGMA, written for clarity rather than speed.

    Repeatedly merge the two clusters with the smallest mean cross-distance,
    stopping once that mean exceeds the cutoff.
    """
    clusters = [[i] for i in range(matrix.shape[0])]
    while True:
        best = None
        for i, j in itertools.combinations(range(len(clusters)), 2):
            mean = float(np.mean([matrix[a][b]
                                  for a in clusters[i] for b in clusters[j]]))
            if mean <= cutoff and (best is None or mean < best[0]):
                best = (mean, i, j)
        if best is None:
            return sorted(sorted(c) for c in clusters)
        _mean, i, j = best
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]


@pytest.mark.parametrize("cutoff", [0.15, 0.30, 0.45])
def test_agglomeration_matches_textbook_upgma(cutoff):
    """
    The safety net for the heap-based clustering.

    Given every pair (radius 1.0 leaves nothing uncomputed, so no distance is
    ever estimated), it must reproduce plain UPGMA exactly.
    """
    n = 36
    rng = np.random.default_rng(5)
    features = rng.random((n, engine.FEATURE_DIMS)).astype(np.float32) * 0.03
    bits = rng.integers(0, 2 ** 63, size=(n, engine.STRUCT_WORDS),
                        dtype=np.int64).astype(np.uint64)

    left, right, dist = engine._all_pairs_within(features, bits, 1.0)
    assert left.size == n * (n - 1) // 2, "radius 1.0 must keep every pair"

    matrix = np.zeros((n, n))
    matrix[left, right] = dist
    matrix[right, left] = dist

    got = sorted(engine._agglomerate(n, left, right, dist, cutoff, 1.0,
                                     None, None))
    assert got == reference_average_linkage(matrix, cutoff)


def test_average_linkage_refuses_to_chain():
    """
    A ladder of images, each close to the next but the ends far apart.

    Single linkage - which is what union-find over the near pairs gives you -
    would swallow the whole ladder into one group.  Average linkage must not.
    """
    n = 12
    step = 0.06
    features = np.zeros((n, engine.FEATURE_DIMS), dtype=np.float32)
    features[:, 0] = np.arange(n) * step
    bits = np.zeros((n, engine.STRUCT_WORDS), dtype=np.uint64)

    left, right, dist = engine._all_pairs_within(features, bits, 1.0)
    clusters = engine._agglomerate(n, left, right, dist, step * 1.2, 1.0,
                                   None, None)

    assert len(clusters) > 1, "the ladder chained into a single group"
    assert max(len(c) for c in clusters) < n
    for cluster in clusters:                    # every group stays contiguous
        assert cluster == list(range(min(cluster), max(cluster) + 1))


# --------------------------------------------------------------------------- #
#  Grouping end to end
# --------------------------------------------------------------------------- #

def test_groups_variants_together_and_keeps_strangers_apart(tmp_path):
    paths = []
    for family in range(3):
        original = scene(str(tmp_path / f"f{family}.png"),
                         seed=100 + family, size=400)
        paths.append(original)
        with Image.open(original) as image:
            rgb = image.convert("RGB")
            for copy in range(2):
                target = str(tmp_path / f"f{family}_v{copy}.jpg")
                rgb.resize((400 - copy * 60, 400 - copy * 60),
                           Image.Resampling.LANCZOS).save(target, quality=80)
                paths.append(target)

    infos, errors = scan_images(paths)
    assert not errors
    groups = group_images(infos, 85.0)

    real = [g for g in groups if len(g) > 1]
    assert len(real) == 3, [[os.path.basename(i.path) for i in g]
                            for g in groups]
    for group in real:
        stems = {os.path.basename(i.path).split("_")[0].split(".")[0]
                 for i in group}
        assert len(stems) == 1, [i.path for i in group]
        assert len(group) == 3


def test_group_and_index_numbering_produces_the_new_names(tmp_path):
    paths = []
    for family in range(2):
        original = scene(str(tmp_path / f"s{family}.png"), seed=7 + family)
        paths.append(original)
        with Image.open(original) as image:
            image.convert("RGB").save(tmp_path / f"s{family}copy.jpg",
                                      quality=92)
        paths.append(str(tmp_path / f"s{family}copy.jpg"))

    groups = group_images(scan_images(paths)[0], 85.0)
    names = [info.new_name for group in groups for info in group]

    assert sorted(names) == sorted(["1_1.png", "1_2.jpg", "2_1.png", "2_2.jpg"])
    for number, group in enumerate(groups, start=1):
        assert [i.group for i in group] == [number] * len(group)
        assert [i.index for i in group] == list(range(1, len(group) + 1))
        assert group[0].match == 100.0


def test_extension_is_lower_cased(tmp_path):
    upper = scene(str(tmp_path / "SHOUT.JPG"), seed=4)
    assert upper.endswith(".JPG")
    groups = group_images(scan_images([upper])[0], 80.0)
    assert groups[0][0].new_name == "1_1.jpg"


def test_looser_threshold_never_makes_more_groups(tmp_path):
    paths = [scene(str(tmp_path / f"n{i}.png"), seed=i) for i in range(12)]
    infos = scan_images(paths)[0]
    counts = [len(group_images(infos, pct)) for pct in (95, 85, 75, 65, 55)]
    assert counts == sorted(counts, reverse=True), counts


@pytest.mark.parametrize("choice,expected_groups", [
    (UNGROUPED_OPTIONS[0], 3),      # one group each
    (UNGROUPED_OPTIONS[1], 1),      # all in one
    (UNGROUPED_OPTIONS[2], 0),      # dropped
])
def test_ungrouped_handling(tmp_path, choice, expected_groups):
    paths = [scene(str(tmp_path / f"u{i}.png"), seed=200 + i * 37)
             for i in range(3)]
    groups = group_images(scan_images(paths)[0], 95.0, ungrouped=choice)
    assert len(groups) == expected_groups
    assert sum(len(g) for g in groups) == (0 if expected_groups == 0 else 3)


def test_grouping_is_deterministic(tmp_path):
    paths = [scene(str(tmp_path / f"d{i}.png"), seed=i // 2, size=260)
             for i in range(8)]
    infos = scan_images(paths)[0]
    first = [[i.path for i in g] for g in group_images(infos, 80.0)]
    second = [[i.path for i in g] for g in group_images(
        scan_images(paths)[0], 80.0)]
    assert first == second


def test_empty_and_single_inputs_do_not_explode(tmp_path):
    assert group_images([], 80.0) == []
    only = scan_images([scene(str(tmp_path / "one.png"), seed=1)])[0]
    groups = group_images(only, 80.0)
    assert len(groups) == 1 and groups[0][0].new_name == "1_1.png"


# --------------------------------------------------------------------------- #
#  Discovery
# --------------------------------------------------------------------------- #

def test_collect_files_spans_folders_and_deduplicates(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    nested = left / "inner"
    for folder in (left, right, nested):
        folder.mkdir(parents=True, exist_ok=True)
    scene(str(left / "a.png"), seed=1)
    scene(str(nested / "b.png"), seed=2)
    scene(str(right / "c.png"), seed=3)
    (right / "notes.txt").write_text("ignored")

    found = sorted(os.path.basename(p)
                   for p in collect_files([str(left), str(right)], True))
    assert found == ["a.png", "b.png", "c.png"]

    shallow = sorted(os.path.basename(p)
                     for p in collect_files([str(left)], False))
    assert shallow == ["a.png"]

    # Overlapping selections must not yield the same file twice.
    twice = list(collect_files([str(left), str(nested)], True))
    assert len(twice) == len(set(twice)) == 2


# --------------------------------------------------------------------------- #
#  Organising
# --------------------------------------------------------------------------- #

def _grouped(tmp_path, families: int = 2):
    paths = []
    for family in range(families):
        original = scene(str(tmp_path / f"g{family}.png"), seed=50 + family)
        paths.append(original)
        with Image.open(original) as image:
            image.convert("RGB").save(tmp_path / f"g{family}b.jpg", quality=90)
        paths.append(str(tmp_path / f"g{family}b.jpg"))
    return group_images(scan_images(paths)[0], 85.0)


def test_organise_copies_under_the_new_names(tmp_path):
    groups = _grouped(tmp_path)
    out = tmp_path / "out"
    out.mkdir()

    copied, written, errors = organise(groups, str(out))
    assert not errors
    assert copied == sum(len(g) for g in groups)
    assert written > 0

    expected = sorted(info.new_name for group in groups for info in group)
    assert sorted(os.listdir(out)) == expected
    # Copies, never moves: every source is still where it was.
    for group in groups:
        for info in group:
            assert os.path.exists(info.path)


def test_organise_can_split_into_subfolders(tmp_path):
    groups = _grouped(tmp_path)
    out = tmp_path / "out"
    copied, _written, errors = organise(groups, str(out), subfolders=True)
    assert not errors and copied == 4
    assert sorted(os.listdir(out)) == ["group_1", "group_2"]
    assert sorted(os.listdir(out / "group_1")) == ["1_1.png", "1_2.jpg"]


def test_organise_reports_unreadable_sources(tmp_path):
    groups = _grouped(tmp_path, families=1)
    groups[0][1].path = str(tmp_path / "vanished.jpg")
    out = tmp_path / "out"
    copied, _written, errors = organise(groups, str(out))
    assert copied == 1
    assert len(errors) == 1 and "vanished" in errors[0][0]


def test_organise_progress_counts_every_file(tmp_path):
    groups = _grouped(tmp_path)
    seen: list[tuple[int, int]] = []
    organise(groups, str(tmp_path / "out"), progress=lambda d, t: seen.append(
        (d, t)))
    assert seen[-1] == (4, 4)
    assert [d for d, _t in seen] == [1, 2, 3, 4]


def test_destination_conflicts_are_detected(tmp_path):
    groups = _grouped(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    assert destination_conflicts(groups, str(out)) == []
    (out / "1_1.png").write_bytes(b"x")
    assert destination_conflicts(groups, str(out)) == ["1_1.png"]


def test_is_inside_catches_nested_destinations(tmp_path):
    outer = tmp_path / "photos"
    inner = outer / "sorted"
    inner.mkdir(parents=True)
    assert is_inside(str(inner), str(outer))
    assert is_inside(str(outer), str(outer))
    assert not is_inside(str(outer), str(inner))
    assert not is_inside(str(tmp_path / "elsewhere"), str(outer))


# --------------------------------------------------------------------------- #
#  Cache
# --------------------------------------------------------------------------- #

def test_cache_round_trips_features(tmp_path):
    cache = FeatureCache(str(tmp_path / "c.sqlite3"))
    assert cache.available

    rng = np.random.default_rng(1)
    features = rng.random(engine.FEATURE_DIMS).astype(np.float32)
    bits = rng.integers(0, 255, size=engine.STRUCT_BITS // 8,
                        dtype=np.uint8).tobytes()
    key = os.path.normcase(os.path.abspath("k"))
    cache.store([CachedFeatures(key=key, size=10, mtime_ns=20, bits=bits,
                                features=quantise(features))])

    loaded = cache.load(["k"])
    assert list(loaded) == [key]
    row = loaded[key]
    assert row.size == 10 and row.mtime_ns == 20 and row.bits == bits
    assert np.max(np.abs(dequantise(row.features) - features)) < 1.0 / 255.0


def test_cache_speeds_up_a_rescan_without_changing_it(tmp_path):
    paths = [scene(str(tmp_path / f"c{i}.png"), seed=i) for i in range(6)]
    cache = FeatureCache(str(tmp_path / "c.sqlite3"))

    cold, _errors = scan_images(paths, cache=cache)
    assert cache.count() == len(paths)
    warm, _errors = scan_images(paths, cache=cache)

    for a, b in zip(cold, warm, strict=True):
        assert a.path == b.path
        assert np.array_equal(a.bits, b.bits)
        # The cache stores quantised features, so allow one quantisation step.
        assert np.max(np.abs(a.features - b.features)) <= 1.0 / 255.0


def test_stale_cache_rows_are_ignored(tmp_path):
    target = scene(str(tmp_path / "s.png"), seed=1)
    cache = FeatureCache(str(tmp_path / "c.sqlite3"))
    scan_images([target], cache=cache)

    scene(target, seed=42)                      # same path, different picture
    os.utime(target, (1_000_000, 1_000_000))
    fresh = scan_images([target], cache=cache)[0][0]
    direct = scan_images([target])[0][0]
    assert np.array_equal(fresh.bits, direct.bits)


def test_a_broken_cache_never_breaks_a_scan(tmp_path):
    broken = tmp_path / "broken.sqlite3"
    broken.write_bytes(b"this is not a database")
    cache = FeatureCache(str(broken))

    paths = [scene(str(tmp_path / "b.png"), seed=1)]
    infos, errors = scan_images(paths, cache=cache)
    assert not errors and len(infos) == 1
    assert cache.load(paths) == {}
    cache.store([])
    cache.clear()


def test_cache_clear_empties_it(tmp_path):
    paths = [scene(str(tmp_path / f"x{i}.png"), seed=i) for i in range(3)]
    cache = FeatureCache(str(tmp_path / "c.sqlite3"))
    scan_images(paths, cache=cache)
    assert cache.count() == 3
    cache.clear()
    assert cache.count() == 0
    # A VACUUM that ran inside the delete's transaction would roll it back.
    assert cache.load(paths) == {}


# --------------------------------------------------------------------------- #
#  Scanning edge cases
# --------------------------------------------------------------------------- #

def test_unreadable_files_are_reported_not_raised(tmp_path):
    good = scene(str(tmp_path / "good.png"), seed=1)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"definitely not a PNG")
    missing = str(tmp_path / "gone.png")

    infos, errors = scan_images([good, str(bad), missing])
    assert [i.path for i in infos] == [good]
    assert sorted(os.path.basename(p) for p, _m in errors) == ["bad.png",
                                                               "gone.png"]


def test_exif_orientation_is_honoured(tmp_path):
    """A rotated photo should match the copy that is already the right way up."""
    upright = str(tmp_path / "upright.jpg")
    with Image.open(scene(str(tmp_path / "src.png"), seed=6, size=400)) as im:
        # Orientation 8 is what exif_transpose applies as ROTATE_90, so the
        # already-upright copy has to be rotated the same way, not the other.
        im.convert("RGB").transpose(Image.Transpose.ROTATE_90).save(
            upright, quality=95)

    sideways = str(tmp_path / "sideways.jpg")
    with Image.open(str(tmp_path / "src.png")) as im:
        exif = Image.Exif()
        exif[274] = 8                            # Orientation: rotate 90 CCW
        im.convert("RGB").save(sideways, quality=95, exif=exif)

    assert similarity(describe(upright), describe(sideways)) > 88.0


def test_cancelling_a_scan_stops_it(tmp_path):
    import threading

    paths = [scene(str(tmp_path / f"k{i}.png"), seed=i) for i in range(4)]
    cancel = threading.Event()
    cancel.set()
    infos, _errors = scan_images(paths, cancel=cancel)
    assert len(infos) < len(paths)


def test_cancelling_grouping_returns_nothing(tmp_path):
    import threading

    paths = [scene(str(tmp_path / f"c{i}.png"), seed=i) for i in range(4)]
    cancel = threading.Event()
    cancel.set()
    assert group_images(scan_images(paths)[0], 80.0, cancel=cancel) == []


# --------------------------------------------------------------------------- #
#  Settings
# --------------------------------------------------------------------------- #

def test_settings_round_trip(tmp_path):
    target = str(tmp_path / "settings.json")
    Settings(recursive=False, threshold=72, use_cache=False,
             destination="D:/out", subfolders=True, geometry="900x700+10+10",
             maximized=True, sash=321).save(target)
    loaded = Settings.load(target)
    assert loaded.recursive is False
    assert loaded.threshold == 72
    assert loaded.destination == "D:/out"
    assert loaded.subfolders is True
    assert loaded.maximized is True
    assert loaded.sash == 321


def test_settings_ignore_wrongly_typed_fields(tmp_path):
    target = tmp_path / "settings.json"
    target.write_text('{"recursive": "yes", "threshold": "high",'
                      ' "sash": true, "destination": 5, "subfolders": 1}')
    loaded = Settings.load(str(target))
    assert loaded.recursive is Settings().recursive
    assert loaded.threshold == Settings().threshold
    assert loaded.sash == Settings().sash
    assert loaded.destination == Settings().destination
    assert loaded.subfolders is Settings().subfolders


def test_settings_survive_a_corrupt_file(tmp_path):
    target = tmp_path / "settings.json"
    target.write_text("{not json at all")
    assert Settings.load(str(target)) == Settings()
    assert Settings.load(str(tmp_path / "absent.json")) == Settings()


def test_threshold_is_clamped_to_the_slider_range(tmp_path):
    target = tmp_path / "settings.json"
    target.write_text('{"threshold": 500}')
    assert Settings.load(str(target)).threshold <= 100
    target.write_text('{"threshold": -20}')
    assert Settings.load(str(target)).threshold >= 40
