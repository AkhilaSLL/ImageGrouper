"""
Image Grouper
=============

Groups visually similar images and copies them into one destination folder,
renamed ``<group>_<number><ext>`` - a way to reorganise a scattered image dump
into runs of pictures that belong together.

Similarity is judged from the pixels only: shape (dHash + a 256-bit DCT hash),
colour layout (an 8x8 CIELAB thumbnail), local detail, and colour palette.
File size, resolution and metadata are deliberately never consulted, because
none of them survive a resize or a re-encode.

Nothing is ever moved or deleted - sources are only read.
"""

__version__ = "1.0.0"

__all__ = ["__version__", "main"]


def main() -> None:
    """Launch the GUI.  Imported lazily so ``import imggroup`` stays cheap."""
    from .app import main as _main

    _main()
