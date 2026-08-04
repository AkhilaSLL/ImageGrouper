"""
Image embeddings from DINOv2-small, run on the CPU through onnxruntime.

This replaced a hand-crafted descriptor (perceptual hashes + CIELAB colour
layout + a local-detail map + palette histograms) that failed on real dumps in
both directions at once: it missed photographs of the same place shot from a
different angle, and it happily grouped unrelated pictures that merely shared a
colour cast.  That is not a tuning failure.  Global pixel statistics describe
how an image *looks as a grid of numbers*, and "do these belong in the same
album" is a question about what is *in* it, which no amount of reweighting
those statistics can answer.

The choice of model and recipe here is measured, not assumed - see ``probe/``:

* **DINOv2-small over CLIP ViT-B/32.**  On albums of one scene shot several
  ways, DINOv2 scored mAP 0.653 against CLIP's 0.549 and the old descriptor's
  0.472, and separated same-album from unrelated pairs by 0.483 where the
  others managed 0.120 and 0.115.  That separation is what makes a threshold
  mean anything.  It is also four times smaller.
* **uint8 over fp32/fp16/q4.**  uint8 matches fp32 to within noise (mAP 0.654
  vs 0.648) at a quarter of the size.  q4 is genuinely worse (0.630) and not
  worth the 7 MB it saves.
* **Pad to square, don't centre-crop.**  The standard recipe crops away 44% of
  a 16:9 photograph.  Keeping the whole frame scored better (mAP 0.654 vs
  0.645) *and* ran faster.
* **No centring.**  Subtracting the collection mean looked like a clear win
  (AUC 0.9589 -> 0.9698) but that was measured on the same images the mean came
  from; on held-out albums the gain vanished (0.9818 vs 0.9817).  It also has a
  fatal edge case - with two images the mean sits exactly between them, so the
  centred vectors are exact opposites and every pair scores 0%.

Resolution is fixed at 224px: the exported graph bakes in 257 position
embeddings, so anything else fails outright in the position-embedding add.
"""

from __future__ import annotations

import threading

import numpy as np
from PIL import Image

from .resources import asset_path

MODEL_FILE = "dinov2-small-uint8.onnx"
IMAGE_SIZE = 224
EMBED_DIM = 384
BATCH_SIZE = 16

_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


class EmbedderUnavailable(RuntimeError):
    """The ONNX model or runtime could not be loaded."""


def preprocess(image: Image.Image) -> np.ndarray:
    """
    RGB image -> ``(3, 224, 224)`` float32, ready for the model.

    The whole frame is scaled to fit and the remainder padded, rather than
    centre-cropped.  Padding happens *after* normalisation so the border is
    exactly zero - i.e. exactly the dataset mean, contributing no signal.
    Padding with an 8-bit approximation of the mean instead leaves a residual
    of ~0.008, which is harmless but not something worth leaving to chance.
    """
    if image.mode != "RGB":       # greyscale or RGBA would break the broadcast
        image = image.convert("RGB")
    scale = IMAGE_SIZE / max(image.width, image.height)
    width = min(IMAGE_SIZE, max(1, round(image.width * scale)))
    height = min(IMAGE_SIZE, max(1, round(image.height * scale)))
    resized = image.resize((width, height), Image.Resampling.BICUBIC)

    array = (np.asarray(resized, dtype=np.float32) / 255.0 - _MEAN) / _STD
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    top, left = (IMAGE_SIZE - height) // 2, (IMAGE_SIZE - width) // 2
    canvas[top:top + height, left:left + width] = array
    return canvas.transpose(2, 0, 1)


class Embedder:
    """
    Lazily-loaded ONNX session producing one unit-length vector per image.

    The session is built on first use rather than at import, so the GUI starts
    instantly and a missing model only breaks scanning rather than launching.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or asset_path(MODEL_FILE)
        self._session = None
        self._input = ""
        self._lock = threading.Lock()

    def _ensure(self):                                     # noqa: ANN202
        if self._session is not None:
            return self._session
        with self._lock:
            if self._session is not None:
                return self._session
            try:
                import onnxruntime as ort
            except Exception as exc:                       # pragma: no cover
                raise EmbedderUnavailable(
                    f"onnxruntime is not available: {exc}") from exc
            try:
                options = ort.SessionOptions()
                options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
                options.log_severity_level = 3     # keep ORT off the console
                session = ort.InferenceSession(
                    self.model_path, options,
                    providers=["CPUExecutionProvider"])
            except Exception as exc:
                raise EmbedderUnavailable(
                    f"could not load {self.model_path}: {exc}") from exc
            self._input = session.get_inputs()[0].name
            self._session = session
            return session

    def embed(self, batch: np.ndarray) -> np.ndarray:
        """``(n, 3, 224, 224)`` -> ``(n, EMBED_DIM)`` unit-length float32."""
        session = self._ensure()
        tokens = session.run(None, {self._input:
                                    np.ascontiguousarray(batch, np.float32)})[0]
        # Token 0 is the CLS summary; the remaining 256 are per-patch.  Pooling
        # the patches instead, or concatenating both, measured slightly worse.
        vectors = np.asarray(tokens[:, 0], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-9)


_shared: Embedder | None = None
_shared_lock = threading.Lock()


def shared() -> Embedder:
    """The process-wide embedder, so the model is loaded at most once."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = Embedder()
        return _shared
