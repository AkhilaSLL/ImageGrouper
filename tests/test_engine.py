"""
Headless tests for the grouping engine, cache and settings.  No GUI is
imported anywhere in here, so the whole suite runs without a display.

A note on the fixtures, because it is not obvious and it caught me out: the
generators below deliberately produce pictures of genuinely *different kinds*
(a page of text, a bar chart, a sky, a noise texture...).  An earlier version
used one abstract generator with different random seeds, which was fine against
a colour-statistics descriptor but is meaningless against an embedding - six
pictures of randomly coloured circles scored 81-86% alike, because to the model
they honestly are the same sort of thing.  Same kind, different seed is the
"same album" case here; different kind is the "unrelated" case.
"""

from __future__ import annotations

import itertools
import os
import random

import numpy as np
import pytest
from PIL import Image, ImageDraw

from imggroup import embedder, engine
from imggroup.cache import SCHEMA_VERSION, FeatureCache
from imggroup.engine import (
    COSINE_FLOOR,
    UNGROUPED_OPTIONS,
    CachedFeatures,
    ImgInfo,
    collect_files,
    destination_conflicts,
    distance_from_cosine,
    group_images,
    is_inside,
    organise,
    pack_features,
    scan_images,
    similarity,
    unpack_features,
)
from imggroup.settings import DEFAULT_THRESHOLD, Settings

# --------------------------------------------------------------------------- #
#  Image fixtures - one function per *kind* of picture
# --------------------------------------------------------------------------- #

def text_page(path: str, seed: int = 0) -> str:
    """A scanned page: white, with text blocks in varying places."""
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


def bar_chart(path: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    image = Image.new("RGB", (520, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.line((50, 350, 500, 350), fill="black", width=3)
    draw.line((50, 30, 50, 350), fill="black", width=3)
    for bar in range(8):
        x = 70 + bar * 52
        draw.rectangle((x, 350 - rng.randint(40, 300), x + 34, 350),
                       fill=(60, 110, 190))
    image.save(path)
    return path


def colour_bands(path: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    image = Image.new("RGB", (420, 560))
    draw = ImageDraw.Draw(image)
    for band in range(6):
        draw.rectangle((0, band * 94, 420, band * 94 + 94),
                       fill=(rng.randrange(256), rng.randrange(256),
                             rng.randrange(256)))
    image.save(path)
    return path


def sky_gradient(path: str, seed: int = 0) -> str:
    """Stands in for a photograph: sky over a dark horizon band."""
    rng = random.Random(seed)
    top = (rng.randrange(20, 90), rng.randrange(80, 160),
           rng.randrange(170, 255))
    bottom = (rng.randrange(200, 255), rng.randrange(150, 220),
              rng.randrange(90, 160))
    image = Image.new("RGB", (520, 380))
    draw = ImageDraw.Draw(image)
    for y in range(380):
        t = y / 379
        draw.line([(0, y), (520, y)],
                  fill=tuple(int(top[i] + (bottom[i] - top[i]) * t)
                             for i in range(3)))
    draw.rectangle((0, 300, 520, 380), fill=(30 + seed % 20, 60, 35))
    image.save(path)
    return path


def noise_texture(path: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    image = Image.new("RGB", (400, 400))
    pixels = image.load()
    assert pixels is not None
    for y in range(0, 400, 4):
        for x in range(0, 400, 4):
            value = rng.randrange(256)
            for dy in range(4):
                for dx in range(4):
                    pixels[x + dx, y + dy] = (value, value, value)
    image.save(path)
    return path


# Ordered so KINDS[:n] is always a well-separated selection.
KINDS = (text_page, bar_chart, sky_gradient, noise_texture, colour_bands)


def describe(path: str) -> ImgInfo:
    infos, errors = scan_images([path])
    assert not errors, errors
    return infos[0]


def variants(tmp_path, maker, stem: str, count: int = 3) -> list[str]:
    """One picture saved several ways - the "same photo, edited" case."""
    original = maker(str(tmp_path / f"{stem}.png"), seed=1)
    paths = [original]
    with Image.open(original) as image:
        rgb = image.convert("RGB")
        for copy in range(1, count):
            target = str(tmp_path / f"{stem}_v{copy}.jpg")
            rgb.resize((int(rgb.width * (1 - 0.12 * copy)),
                        int(rgb.height * (1 - 0.12 * copy))),
                       Image.Resampling.LANCZOS).save(target, quality=85)
            paths.append(target)
    return paths


# --------------------------------------------------------------------------- #
#  The similarity scale
# --------------------------------------------------------------------------- #

def test_distance_mapping_has_no_constant_floor():
    """
    The regression test for the bug that made v1 useless.

    v1 gave a perceptual-hash block 26% of the squared distance, but measured
    over real photographs that block sat between 0.64 and 0.75 for *every* pair
    of distinct pictures - near-constant.  A constant inside a square root is a
    floor on the distance and therefore a ceiling on similarity:

        d >= sqrt(0.26 * 0.687^2) = 0.350  ->  similarity <= 65%

    so nothing but a byte-identical file could reach the default setting.  The
    mapping must span the full range and reach 0 exactly.
    """
    assert distance_from_cosine(1.0) == pytest.approx(0.0)
    assert distance_from_cosine(COSINE_FLOOR) == pytest.approx(1.0)
    assert distance_from_cosine(0.0) == pytest.approx(1.0)      # clipped
    assert distance_from_cosine(-1.0) == pytest.approx(1.0)

    # Strictly decreasing in between, with no dead zone at the top.
    cosines = np.linspace(COSINE_FLOOR, 1.0, 21)
    scores = 100.0 * (1.0 - distance_from_cosine(cosines))
    assert np.all(np.diff(scores) > 0)
    assert scores[0] == pytest.approx(0.0)
    assert scores[-1] == pytest.approx(100.0)


def test_squared_limit_agrees_with_the_distance_mapping():
    """
    The BLAS search filters on squared Euclidean distance while the confirm
    step uses the cosine mapping.  If those two disagree the search silently
    drops real pairs, so pin the conversion at the boundary.
    """
    for radius in (0.05, 0.2, 0.35, 0.5, 0.8):
        limit = engine.squared_limit(radius)
        # ||a-b||^2 == limit is exactly the pair whose distance == radius.
        cosine = 1.0 - limit / 2.0
        assert float(distance_from_cosine(cosine)) == pytest.approx(
            radius, abs=1e-9)


def test_radius_one_really_means_every_pair():
    """
    Everything at or below COSINE_FLOOR clips to distance exactly 1.0, so a
    radius of 1 has to include opposite vectors too - not just those down to
    the floor.  Taking the algebraic limit there would drop the very pairs the
    caller asked for, and the UPGMA cross-check depends on getting all of them.
    """
    features = np.zeros((4, engine.EMBED_DIM), dtype=np.float32)
    features[0, 0], features[1, 0] = 1.0, -1.0          # exactly opposite
    features[2, 1], features[3, 2] = 1.0, 1.0           # orthogonal to both

    left, _right, dist = engine.near_pairs(features, 1.0)
    assert left.size == 6, "radius 1.0 dropped pairs below the cosine floor"
    assert np.allclose(dist, 1.0)


def test_identical_files_are_100_percent(tmp_path):
    first = text_page(str(tmp_path / "a.png"), seed=1)
    second = str(tmp_path / "b.png")
    with Image.open(first) as image:
        image.save(second)
    assert similarity(describe(first), describe(second)) == pytest.approx(
        100.0, abs=0.05)


def test_a_different_file_can_still_reach_the_top_of_the_slider(tmp_path):
    """
    The other half of the ceiling regression: not merely that the *formula*
    reaches 100, but that two genuinely different files can score above the
    default threshold.  Under v1 the best any pair of distinct pictures could
    manage was 65%, which is what left almost everything in a group of one.
    """
    original = sky_gradient(str(tmp_path / "orig.png"), seed=3)
    edited = str(tmp_path / "edited.jpg")
    with Image.open(original) as image:
        image.convert("RGB").resize((430, 314),
                                    Image.Resampling.LANCZOS).save(
            edited, quality=78)

    assert os.path.getsize(original) != os.path.getsize(edited)
    score = similarity(describe(original), describe(edited))
    assert score > DEFAULT_THRESHOLD, f"edited copy only scored {score:.1f}%"
    assert score < 100.0, "these are not the same bytes; 100% would be wrong"


def test_reencoding_and_resizing_stays_highly_similar(tmp_path):
    original = bar_chart(str(tmp_path / "orig.png"), seed=2)
    with Image.open(original) as image:
        rgb = image.convert("RGB")
        rgb.resize((260, 200), Image.Resampling.LANCZOS).save(
            tmp_path / "half.jpg", quality=70)
        rgb.save(tmp_path / "q35.jpg", quality=35)

    base = describe(original)
    for name in ("half.jpg", "q35.jpg"):
        score = similarity(base, describe(str(tmp_path / name)))
        assert score > 88.0, f"{name} scored {score:.1f}%"


def test_unrelated_pictures_score_below_edits_of_one(tmp_path):
    """
    The property that makes a threshold mean anything, stated as a separation
    rather than a magic constant: the *worst* pair of edits of one picture must
    still outscore the *best* pair of unrelated ones.  Under v1 this was false
    - edits topped out at 65% while unrelated photographs reached 73%, which is
    precisely why no setting of the slider worked.
    """
    related: list[float] = []
    for number, maker in enumerate(KINDS):
        family = [describe(p) for p in variants(tmp_path, maker, f"r{number}")]
        related += [similarity(a, b)
                    for a, b in itertools.combinations(family, 2)]

    singles = [describe(maker(str(tmp_path / f"u{number}.png"), seed=5))
               for number, maker in enumerate(KINDS)]
    unrelated = [similarity(a, b)
                 for a, b in itertools.combinations(singles, 2)]

    assert min(related) > max(unrelated), (
        f"edits of one picture bottom out at {min(related):.1f}% while "
        f"unrelated pictures reach {max(unrelated):.1f}% - no threshold can "
        "separate them")


# --------------------------------------------------------------------------- #
#  Embedding
# --------------------------------------------------------------------------- #

def test_embedding_is_unit_length(tmp_path):
    info = describe(text_page(str(tmp_path / "e.png"), seed=1))
    assert info.features.shape == (embedder.EMBED_DIM,)
    assert info.features.dtype == np.float32
    assert float(np.linalg.norm(info.features)) == pytest.approx(1.0, abs=1e-5)


def test_preprocess_keeps_the_whole_frame():
    """
    The standard recipe resizes the shortest edge and centre-crops, which
    throws away 44% of a 16:9 picture.  Padding measured better *and* faster,
    so a wide picture must survive end to end instead of being cropped.
    """
    wide = Image.new("RGB", (640, 200), (200, 30, 30))
    draw = ImageDraw.Draw(wide)
    draw.rectangle((0, 0, 60, 199), fill=(0, 0, 255))        # far left
    draw.rectangle((580, 0, 639, 199), fill=(0, 255, 0))     # far right

    array = embedder.preprocess(wide)
    assert array.shape == (3, embedder.IMAGE_SIZE, embedder.IMAGE_SIZE)

    # 640x200 fitted into 224 leaves a 224x70 strip centred vertically, so the
    # bands above and below it are padding - applied after normalisation, so
    # exactly zero and contributing no signal at all.
    assert np.abs(array[:, :60, :]).max() == 0.0, "padding is not neutral"
    assert np.abs(array[:, -60:, :]).max() == 0.0, "padding is not neutral"

    middle = array[:, 100:120, :]
    assert np.abs(middle).max() > 0.5, "the picture itself went missing"
    # Both edge markers survive: blue dominates on the left, green on the right.
    assert middle[2, :, :8].mean() > middle[1, :, :8].mean()
    assert middle[1, :, -8:].mean() > middle[2, :, -8:].mean()


def test_float16_round_trip_is_faithful():
    rng = np.random.default_rng(4)
    values = rng.normal(size=embedder.EMBED_DIM).astype(np.float32)
    values /= np.linalg.norm(values)
    back = unpack_features(pack_features(values))
    assert back.shape == values.shape
    # Far below the gap that decides any grouping.
    assert float(np.abs(back - values).max()) < 1e-3
    assert float(back @ values) == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------- #
#  Near-pair search
# --------------------------------------------------------------------------- #

def _synthetic_population(n: int, seed: int = 0,
                          spread: float = 0.35) -> np.ndarray:
    """
    Clustered unit vectors, standing in for embeddings.

    They have to be genuinely close together.  Two random directions in 384
    dimensions are near-orthogonal, and every cosine at or below COSINE_FLOOR
    clips to distance 1.0 - so a uniform sample would make every pair exactly
    equidistant and leave the checks below comparing two identical sets of
    nothing.  Build them around a shared anchor instead.
    """
    rng = np.random.default_rng(seed)
    anchor = rng.normal(size=engine.EMBED_DIM)
    centres = anchor + rng.normal(0, spread,
                                  (max(2, n // 25), engine.EMBED_DIM))
    pick = rng.integers(0, centres.shape[0], size=n)
    features = centres[pick] + rng.normal(0, spread * 0.25,
                                          (n, engine.EMBED_DIM))
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    return features.astype(np.float32)


def _as_pairs(left, right, dist):
    return {(int(a), int(b)): round(float(d), 9)
            for a, b, d in zip(left, right, dist, strict=True)}


@pytest.mark.parametrize("radius", [0.10, 0.25, 0.40])
def test_expansion_matches_direct_distances(radius):
    """
    The safety net for the one optimisation left in the search.

    The filter expands ``||a-b||^2`` as ``||a||^2 + ||b||^2 - 2a.b`` so that a
    whole block is one BLAS product.  That form loses precision exactly where
    it matters most - near-identical images, where the two large terms cancel -
    so check it against computing the cosine outright.
    """
    n = 400
    features = _synthetic_population(n, seed=7)
    got = _as_pairs(*engine._all_pairs_within(features, radius))

    want = {}
    for i in range(n):
        for j in range(i + 1, n):
            distance = float(distance_from_cosine(
                float(features[i] @ features[j])))
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
    features = _synthetic_population(600, seed=8)
    monkeypatch.setattr(engine, "MAX_PAIRS", 500)
    left, _right, _dist = engine.near_pairs(features, 1.0)
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
    features = _synthetic_population(n, seed=5, spread=0.5)

    left, right, dist = engine._all_pairs_within(features, 1.0)
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
    features = np.zeros((n, engine.EMBED_DIM), dtype=np.float32)
    angles = np.arange(n) * 0.20            # far enough that the ends diverge
    features[:, 0] = np.cos(angles)
    features[:, 1] = np.sin(angles)

    left, right, dist = engine._all_pairs_within(features, 1.0)
    step = float(distance_from_cosine(float(features[0] @ features[1])))
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
    for number, maker in enumerate(KINDS[:3]):
        paths += variants(tmp_path, maker, f"f{number}")

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
    for number, maker in enumerate(KINDS[:2]):
        original = maker(str(tmp_path / f"s{number}.png"), seed=7)
        paths.append(original)
        with Image.open(original) as image:
            image.convert("RGB").save(tmp_path / f"s{number}copy.jpg",
                                      quality=92)
        paths.append(str(tmp_path / f"s{number}copy.jpg"))

    groups = group_images(scan_images(paths)[0], 85.0)
    names = [info.new_name for group in groups for info in group]

    assert sorted(names) == sorted(["1_1.png", "1_2.jpg", "2_1.png", "2_2.jpg"])
    for number, group in enumerate(groups, start=1):
        assert [i.group for i in group] == [number] * len(group)
        assert [i.index for i in group] == list(range(1, len(group) + 1))
        assert group[0].match == 100.0


def test_extension_is_lower_cased(tmp_path):
    upper = str(tmp_path / "SHOUT.JPG")
    bar_chart(upper, seed=4)
    assert upper.endswith(".JPG")
    groups = group_images(scan_images([upper])[0], 80.0)
    assert groups[0][0].new_name == "1_1.jpg"


def test_looser_threshold_never_makes_more_groups(tmp_path):
    paths = [maker(str(tmp_path / f"n{n}_{seed}.png"), seed=seed)
             for n, maker in enumerate(KINDS) for seed in range(2)]
    infos = scan_images(paths)[0]
    counts = [len(group_images(infos, pct)) for pct in (95, 85, 75, 65, 55)]
    assert counts == sorted(counts, reverse=True), counts


@pytest.mark.parametrize("choice,expected_groups", [
    (UNGROUPED_OPTIONS[0], 3),      # one group each
    (UNGROUPED_OPTIONS[1], 1),      # all in one
    (UNGROUPED_OPTIONS[2], 0),      # dropped
])
def test_ungrouped_handling(tmp_path, choice, expected_groups):
    paths = [maker(str(tmp_path / f"u{n}.png"), seed=200 + n * 37)
             for n, maker in enumerate(KINDS[:3])]
    groups = group_images(scan_images(paths)[0], 95.0, ungrouped=choice)
    assert len(groups) == expected_groups
    assert sum(len(g) for g in groups) == (0 if expected_groups == 0 else 3)


def test_grouping_is_deterministic(tmp_path):
    paths = [maker(str(tmp_path / f"d{n}_{seed}.png"), seed=seed)
             for n, maker in enumerate(KINDS[:4]) for seed in range(2)]
    infos = scan_images(paths)[0]
    first = [[i.path for i in g] for g in group_images(infos, 80.0)]
    second = [[i.path for i in g] for g in group_images(
        scan_images(paths)[0], 80.0)]
    assert first == second


def test_empty_and_single_inputs_do_not_explode(tmp_path):
    assert group_images([], 80.0) == []
    only = scan_images([text_page(str(tmp_path / "one.png"), seed=1)])[0]
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
    text_page(str(left / "a.png"), seed=1)
    bar_chart(str(nested / "b.png"), seed=2)
    sky_gradient(str(right / "c.png"), seed=3)
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
    for number, maker in enumerate(KINDS[:families]):
        original = maker(str(tmp_path / f"g{number}.png"), seed=50)
        paths.append(original)
        with Image.open(original) as image:
            image.convert("RGB").save(tmp_path / f"g{number}b.jpg", quality=90)
        paths.append(str(tmp_path / f"g{number}b.jpg"))
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
    features = rng.normal(size=embedder.EMBED_DIM).astype(np.float32)
    features /= np.linalg.norm(features)
    key = os.path.normcase(os.path.abspath("k"))
    cache.store([CachedFeatures(key=key, size=10, mtime_ns=20,
                                features=pack_features(features))])

    loaded = cache.load(["k"])
    assert list(loaded) == [key]
    row = loaded[key]
    assert row.size == 10 and row.mtime_ns == 20
    assert np.max(np.abs(unpack_features(row.features) - features)) < 1e-3


def test_cache_speeds_up_a_rescan_without_changing_it(tmp_path):
    paths = [maker(str(tmp_path / f"c{n}.png"), seed=n)
             for n, maker in enumerate(KINDS)]
    cache = FeatureCache(str(tmp_path / "c.sqlite3"))

    cold, _errors = scan_images(paths, cache=cache)
    assert cache.count() == len(paths)
    warm, _errors = scan_images(paths, cache=cache)

    for a, b in zip(cold, warm, strict=True):
        assert a.path == b.path
        # The cache stores float16, so allow one rounding step.
        assert np.max(np.abs(a.features - b.features)) < 1e-3


def test_stale_cache_rows_are_ignored(tmp_path):
    target = str(tmp_path / "s.png")
    text_page(target, seed=1)
    cache = FeatureCache(str(tmp_path / "c.sqlite3"))
    scan_images([target], cache=cache)

    bar_chart(target, seed=42)                  # same path, different picture
    os.utime(target, (1_000_000, 1_000_000))
    fresh = scan_images([target], cache=cache)[0][0]
    direct = scan_images([target])[0][0]
    assert np.max(np.abs(fresh.features - direct.features)) < 1e-3


def test_rows_from_an_older_schema_are_discarded(tmp_path):
    """
    A cached vector from a previous descriptor is not merely stale - it
    describes the image in a representation that no longer exists.  Reusing it
    would silently group by a mixture of two incompatible feature spaces.
    """
    import sqlite3

    path = str(tmp_path / "old.sqlite3")
    db = sqlite3.connect(path)
    db.executescript(
        "CREATE TABLE features (key TEXT PRIMARY KEY, size INTEGER,"
        " mtime_ns INTEGER, bits BLOB, features BLOB, used_at INTEGER);"
        "INSERT INTO features VALUES ('k', 1, 2, x'00', x'00', 0);"
        "PRAGMA user_version = 1;")
    db.commit()
    db.close()

    cache = FeatureCache(path)
    assert cache.available
    assert cache.count() == 0, "an old-schema row survived the upgrade"

    with sqlite3.connect(path) as check:
        version = int(check.execute("PRAGMA user_version").fetchone()[0])
    assert version == SCHEMA_VERSION


def test_a_broken_cache_never_breaks_a_scan(tmp_path):
    broken = tmp_path / "broken.sqlite3"
    broken.write_bytes(b"this is not a database")
    cache = FeatureCache(str(broken))

    paths = [text_page(str(tmp_path / "b.png"), seed=1)]
    infos, errors = scan_images(paths, cache=cache)
    assert not errors and len(infos) == 1
    assert cache.load(paths) == {}
    cache.store([])
    cache.clear()


def test_cache_clear_empties_it(tmp_path):
    paths = [maker(str(tmp_path / f"x{n}.png"), seed=n)
             for n, maker in enumerate(KINDS[:3])]
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
    good = text_page(str(tmp_path / "good.png"), seed=1)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"definitely not a PNG")
    missing = str(tmp_path / "gone.png")

    infos, errors = scan_images([good, str(bad), missing])
    assert [i.path for i in infos] == [good]
    assert sorted(os.path.basename(p) for p, _m in errors) == ["bad.png",
                                                               "gone.png"]


def test_one_bad_file_does_not_lose_the_rest_of_its_batch(tmp_path):
    """
    Reading is batched now, so a single corrupt file sits in the middle of a
    batch that is embedded in one forward pass.  It must not take its
    neighbours down with it.
    """
    paths = []
    for n, maker in enumerate(KINDS):
        paths.append(maker(str(tmp_path / f"ok{n}.png"), seed=n))
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image at all")
    paths.insert(2, str(broken))

    infos, errors = scan_images(paths)
    assert len(infos) == len(KINDS)
    assert len(errors) == 1 and "broken" in errors[0][0]


def test_exif_orientation_is_honoured(tmp_path):
    """A rotated photo should match the copy that is already the right way up."""
    source = str(tmp_path / "src.png")
    sky_gradient(source, seed=6)

    upright = str(tmp_path / "upright.jpg")
    with Image.open(source) as im:
        # Orientation 8 is what exif_transpose applies as ROTATE_90, so the
        # already-upright copy has to be rotated the same way, not the other.
        im.convert("RGB").transpose(Image.Transpose.ROTATE_90).save(
            upright, quality=95)

    sideways = str(tmp_path / "sideways.jpg")
    with Image.open(source) as im:
        exif = Image.Exif()
        exif[274] = 8                            # Orientation: rotate 90 CCW
        im.convert("RGB").save(sideways, quality=95, exif=exif)

    assert similarity(describe(upright), describe(sideways)) > 88.0


def test_cancelling_a_scan_stops_it(tmp_path):
    import threading

    paths = [maker(str(tmp_path / f"k{n}_{s}.png"), seed=s)
             for n, maker in enumerate(KINDS) for s in range(4)]
    cancel = threading.Event()
    cancel.set()
    infos, _errors = scan_images(paths, cancel=cancel)
    assert len(infos) < len(paths)


def test_cancelling_grouping_returns_nothing(tmp_path):
    import threading

    paths = [maker(str(tmp_path / f"c{n}.png"), seed=n)
             for n, maker in enumerate(KINDS[:4])]
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
