"""
Operating-system integration: opening files and revealing them in the file
manager.

There is deliberately nothing destructive in here.  The app only ever *reads*
the folders it scans and *writes* copies into the destination, so it has no
need for a delete path - and therefore no need for the Recycle Bin dance its
sibling projects carry.
"""

from __future__ import annotations

import os
import subprocess
import sys


def open_file(path: str) -> None:
    """Open ``path`` in the user's default application for that file type."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if sys.platform == "win32":
        os.startfile(path)                                # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def reveal_in_file_manager(path: str) -> None:
    """Show ``path`` selected in Explorer / Finder / the file manager."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    full = os.path.normpath(os.path.abspath(path))
    if sys.platform == "win32":
        # explorer.exe returns 1 even when it succeeds, so the code is ignored.
        subprocess.run(["explorer.exe", f"/select,{full}"], check=False)
    elif sys.platform == "darwin":
        subprocess.run(["open", "-R", full], check=False)
    else:
        subprocess.run(["xdg-open", os.path.dirname(full)], check=False)
