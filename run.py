"""
Launch the app straight from a source checkout: ``python run.py``.

``run.py --selftest [report.txt]`` instead checks that the bundled vision model
loads and embeds.  That is there for the packaged ``.exe``: the interesting
failure in a bundle is onnxruntime's native libraries not surviving it, which
surfaces at session creation rather than at start-up, so launching the window
and screenshotting it would not catch it.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "src"))

from imggroup import main, selftest  # noqa: E402  (needs the path fix above)

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        rest = [a for a in sys.argv[1:] if a != "--selftest"]
        raise SystemExit(selftest(rest[0] if rest else None))
    main()
