"""
Image Grouper
=============

Groups images into albums by what is in them and copies them into one
destination folder, renamed ``<group>_<number><ext>`` - a way to reorganise a
scattered image dump into runs of pictures that belong together.

Similarity is judged from the picture's *content*, via a bundled DINOv2-small
vision model: every image becomes one unit-length 384-number vector, and
similarity is the cosine between two of them.  File size, resolution and
metadata are deliberately never consulted, because none of them survive a
resize or a re-encode.

Nothing is ever moved or deleted - sources are only read.
"""

__version__ = "2.0.0"

__all__ = ["__version__", "main", "selftest"]


def main() -> None:
    """Launch the GUI.  Imported lazily so ``import imggroup`` stays cheap."""
    from .app import main as _main

    _main()


def selftest(report: str | None = None) -> int:
    """
    Check that the bundled model actually loads and embeds.

    This exists for the packaged ``.exe``, where the real risk is not that the
    app fails to start but that onnxruntime's native libraries or the ONNX file
    itself did not survive bundling - which only shows up when a session is
    created, long after the window has appeared.  A screenshot of an idle
    window proves nothing about that.

    Writes a short report and returns a process exit code.  The exe is built
    windowed, so there is no console to print to; pass a file path.
    """
    import io
    import traceback

    out = io.StringIO()
    code = 0
    try:
        import numpy as np
        from PIL import Image

        from .embedder import EMBED_DIM, IMAGE_SIZE, shared
        from .engine import COSINE_FLOOR, distance_from_cosine

        model = shared()
        out.write(f"model     {model.model_path}\n")

        flat = Image.new("RGB", (640, 200), (40, 90, 160))
        noisy = Image.fromarray(np.random.default_rng(0).integers(
            0, 256, size=(240, 240, 3), dtype=np.uint8), "RGB")

        from .embedder import preprocess

        batch = np.stack([preprocess(flat), preprocess(noisy)])
        assert batch.shape == (2, 3, IMAGE_SIZE, IMAGE_SIZE), batch.shape
        vectors = model.embed(batch)

        norms = np.linalg.norm(vectors, axis=1)
        cosine = float(vectors[0] @ vectors[1])
        out.write(f"shape     {vectors.shape} (expected (2, {EMBED_DIM}))\n")
        out.write(f"norms     {norms.round(5).tolist()}\n")
        out.write(f"cosine    {cosine:+.4f}  floor {COSINE_FLOOR}\n")
        out.write(f"similarity {100 * (1 - float(distance_from_cosine(cosine))):.1f}%"
                  "  (two unrelated images)\n")

        assert vectors.shape == (2, EMBED_DIM), "wrong embedding shape"
        assert np.allclose(norms, 1.0, atol=1e-4), "embeddings are not unit"
        assert cosine < 0.95, "two unrelated images came out near-identical"
        out.write("RESULT    OK\n")
    except Exception:
        code = 1
        out.write(traceback.format_exc())
        out.write("RESULT    FAILED\n")

    text = out.getvalue()
    if report:
        with open(report, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text, end="")
    return code
