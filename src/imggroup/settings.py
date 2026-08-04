"""Window and scan settings, persisted between launches."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

from .cache import default_dir
from .engine import UNGROUPED_OPTIONS

FILE_NAME = "settings.json"

MIN_THRESHOLD = 40
MAX_THRESHOLD = 100
# Chosen by running the real clustering over a labelled benchmark at every
# setting and scoring the groups it produced against the true albums: pairwise
# F1 peaks at 70% (0.535, against 0.459 at 75% and 0.446 at 60%).  See
# probe/threshold.py - re-run it rather than nudging this by feel.
DEFAULT_THRESHOLD = 70


@dataclass
class Settings:
    # Source folders are deliberately *not* remembered - each launch starts
    # with an empty list rather than silently re-arming a scan of wherever the
    # app happened to be pointed last time.  The destination is remembered,
    # because that one is usually the same folder every time.
    recursive: bool = True
    threshold: int = DEFAULT_THRESHOLD
    use_cache: bool = True
    destination: str = ""
    subfolders: bool = False
    ungrouped: str = UNGROUPED_OPTIONS[0]
    geometry: str = ""           # the restored (un-maximised) size and position
    maximized: bool = False
    sash: int = 0

    @property
    def path(self) -> str:
        return os.path.join(default_dir(), FILE_NAME)

    @classmethod
    def load(cls, path: str | None = None) -> Settings:
        """Read saved settings.  Anything missing or malformed -> defaults."""
        target = path or os.path.join(default_dir(), FILE_NAME)
        settings = cls()
        try:
            with open(target, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return settings
        if not isinstance(data, dict):
            return settings

        # Read field by field so a hand-edited or older file can never inject
        # a wrong type into the UI; unknown keys are simply ignored.
        for name in ("recursive", "use_cache", "subfolders", "maximized"):
            value = data.get(name)
            if isinstance(value, bool):
                setattr(settings, name, value)
        for name in ("threshold", "sash"):
            value = data.get(name)
            if isinstance(value, int) and not isinstance(value, bool):
                setattr(settings, name, value)
        for name in ("destination", "geometry"):
            value = data.get(name)
            if isinstance(value, str):
                setattr(settings, name, value)
        if data.get("ungrouped") in UNGROUPED_OPTIONS:
            settings.ungrouped = data["ungrouped"]

        settings.threshold = min(MAX_THRESHOLD,
                                 max(MIN_THRESHOLD, settings.threshold))
        return settings

    def save(self, path: str | None = None) -> None:
        """Write settings.  Never raises - failing to save is not a crash."""
        target = path or self.path
        try:
            parent = os.path.dirname(target)
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp = target + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(asdict(self), fh, indent=2)
            os.replace(tmp, target)       # atomic; never a half-written file
        except OSError:
            pass
