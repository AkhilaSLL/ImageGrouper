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


def reveal_command(full: str) -> str:
    """
    Build the exact command line used to reveal ``full`` in Explorer.

    Explorer does **not** parse its command line the way the C runtime does,
    and this is the whole reason this function exists rather than an argument
    list.  Handing ``["explorer.exe", "/select,C:\\a b\\c.jpg"]`` to
    :mod:`subprocess` makes ``list2cmdline`` wrap the argument *as a whole*
    because it contains a space::

        explorer.exe "/select,C:\\a b\\c.jpg"      <- silently wrong

    Explorer cannot read that form: it fails to find a path it recognises and
    falls back to opening the user's default folder (Documents), which looks
    exactly like "Show in Explorer is broken" for every path with a space in
    it - i.e. most of them.  The quotes have to go around the *path* only::

        explorer.exe /select,"C:\\a b\\c.jpg"      <- correct

    Both forms were checked against real Explorer windows before this was
    written; the first opened Documents, the second selected the file.

    Quoting is safe to do by hand here because Windows forbids ``"`` in a path,
    so there is nothing in ``full`` that could terminate the quoted string.
    """
    return f'explorer.exe /select,"{full}"'


def reveal_in_file_manager(path: str) -> None:
    """Show ``path`` selected in Explorer / Finder / the file manager."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    full = os.path.normpath(os.path.abspath(path))
    if sys.platform == "win32":
        # Passed as a string, not a list, so subprocess hands it to
        # CreateProcess verbatim instead of re-quoting it - see
        # reveal_command.  explorer.exe returns 1 even when it succeeds, so
        # the exit code is ignored.
        subprocess.run(reveal_command(full), check=False)
    elif sys.platform == "darwin":
        subprocess.run(["open", "-R", full], check=False)
    else:
        subprocess.run(["xdg-open", os.path.dirname(full)], check=False)
