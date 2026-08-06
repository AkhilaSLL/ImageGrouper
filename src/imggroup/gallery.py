"""Side-by-side preview of one group, captioned with the names to be written."""

from __future__ import annotations

import os
import sys
import tkinter as tk
from collections import OrderedDict
from collections.abc import Callable, Sequence
from tkinter import ttk

from PIL import Image, ImageOps, ImageTk

from .engine import ImgInfo
from .theme import (
    ACCENT,
    BORDER,
    EXEMPLAR_FG,
    FG,
    FG_MUTED,
    FONT,
    FONT_BOLD,
    FONT_MONO,
    WINDOW,
)

THUMB_MIN = 150             # never shrink a thumbnail below this
THUMB_MAX = 560             # width cap so a lone landscape image isn't huge
THUMB_WRAP = 190            # cell size once a group wraps into a contact sheet
SOURCE_CACHE_MAX = 1100     # decoded size kept in memory for fast rescaling
CAPTION_SPACE = 164         # vertical room reserved for the captions
CACHE_BUDGET = 192 << 20    # ~192 MB of decoded previews
RESIZE_DEBOUNCE_MS = 130
FILL_CHUNK = 4              # thumbnails decoded per idle tick
FOLDER_CHARS_MAX = 120      # never show more of a folder path than this
FOLDER_CHARS_MIN = 44       # ...nor less, however narrow the cell gets


def elide_middle(text: str, limit: int) -> str:
    """
    Shorten ``text`` to ``limit`` characters by dropping from the middle.

    The middle is the right thing to drop for a path: the head carries the
    drive and top-level folder, the tail carries the deepest folders, and
    between them those are what tell two source folders apart.  Trimming from
    either end alone throws away one half of that.
    """
    if len(text) <= limit:
        return text
    keep = limit - 1                        # one character goes to the ellipsis
    head = keep // 2
    return f"{text[:head]}…{text[len(text) - (keep - head):]}"


def folder_caption(path: str, box_w: int) -> str:
    """
    The folder ``path`` sits in, trimmed to roughly two lines at ``box_w``.

    Ordinary paths fit whole; the budget only exists so one deeply nested path
    can't wrap into five lines and stretch its cell taller than its neighbours
    - which matters more here than in the sibling projects, because a wrapped
    contact sheet gives every cell only THUMB_WRAP of width to play with.  At
    the 9pt caption font a character averages a little under 5px, so
    ``box_w // 3`` is a conservative two lines' worth.
    """
    budget = min(FOLDER_CHARS_MAX, max(FOLDER_CHARS_MIN, box_w // 3))
    return elide_middle(os.path.dirname(path), budget)


class _SourceCache:
    """
    LRU cache of modestly downsized RGB images.

    Re-decoding a 24 MP JPEG on every window resize is far too slow, but an
    unbounded cache grows by ~3.6 MB per previewed image and will happily eat
    all available memory while browsing a large result set.
    """

    def __init__(self, budget: int = CACHE_BUDGET) -> None:
        self._budget = budget
        self._items: OrderedDict[str, Image.Image] = OrderedDict()
        self._bytes = 0

    @staticmethod
    def _cost(image: Image.Image) -> int:
        return image.width * image.height * 3

    def get(self, path: str) -> Image.Image:
        cached = self._items.get(path)
        if cached is not None:
            self._items.move_to_end(path)
            return cached

        with Image.open(path) as raw:
            # Let libjpeg decode at 1/2, 1/4 or 1/8 scale rather than unpacking
            # 24 MP and throwing most of it away.  It has to come before
            # convert(), which would force the full-resolution decode first.
            raw.draft("RGB", (SOURCE_CACHE_MAX, SOURCE_CACHE_MAX))
            oriented = ImageOps.exif_transpose(raw)
            image = (oriented or raw).convert("RGB")
            image.thumbnail((SOURCE_CACHE_MAX, SOURCE_CACHE_MAX),
                            Image.Resampling.LANCZOS)

        self._items[path] = image
        self._bytes += self._cost(image)
        while self._bytes > self._budget and len(self._items) > 1:
            _, evicted = self._items.popitem(last=False)
            self._bytes -= self._cost(evicted)
        return image

    def clear(self) -> None:
        self._items.clear()
        self._bytes = 0


class Gallery(ttk.LabelFrame):
    """Scrollable, resizable strip of thumbnails for the selected group."""

    def __init__(self, master: tk.Misc, scale: float = 1.0,
                 on_open: Callable[[str], None] | None = None) -> None:
        super().__init__(master, text="  PREVIEW  ")
        self._scale = scale
        self._on_open = on_open
        self._cache = _SourceCache()
        self._group: Sequence[ImgInfo] | None = None
        self._photos: list[ImageTk.PhotoImage] = []
        self._resize_job: str | None = None
        self._fill_job: str | None = None
        self._pending: list[tuple[ImgInfo, tk.Frame]] = []
        self._generation = 0
        self._fitting = False
        self._last_size = (0, 0)
        self._weighted: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())

        self.canvas = tk.Canvas(self, background=WINDOW, highlightthickness=0,
                                bd=0, takefocus=True)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        hsb = ttk.Scrollbar(self, orient="horizontal",
                            command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.canvas.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.inner = tk.Frame(self.canvas, bg=WINDOW)
        self._window = self.canvas.create_window((0, 0), window=self.inner,
                                                 anchor="nw")

        self.canvas.bind("<Configure>", self._on_configure)
        self.inner.bind("<Configure>", lambda _e: self._fit_window())
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)

        self.show_message("Analyse a folder, then select a group to\n"
                          "see its images side by side.")

    # ---- scrolling -------------------------------------------------------- #
    def _bind_wheel(self, _event: tk.Event | None = None) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self.canvas.bind_all("<Shift-MouseWheel>", self._on_shift_wheel)
        if sys.platform.startswith("linux"):    # X11 sends buttons 4/5
            self.canvas.bind_all("<Button-4>", self._on_wheel)
            self.canvas.bind_all("<Button-5>", self._on_wheel)

    def _unbind_wheel(self, _event: tk.Event | None = None) -> None:
        for sequence in ("<MouseWheel>", "<Shift-MouseWheel>",
                         "<Button-4>", "<Button-5>"):
            self.canvas.unbind_all(sequence)

    @staticmethod
    def _wheel_steps(event: tk.Event) -> int:
        if getattr(event, "num", 0) == 4:
            return -1
        if getattr(event, "num", 0) == 5:
            return 1
        return -1 if event.delta > 0 else 1

    def _on_wheel(self, event: tk.Event) -> None:
        # The gallery is a horizontal strip, so a plain wheel scrolls sideways
        # unless the content is actually taller than the viewport.
        if self._overflows_vertically():
            self.canvas.yview_scroll(self._wheel_steps(event), "units")
        else:
            self.canvas.xview_scroll(self._wheel_steps(event) * 3, "units")

    def _on_shift_wheel(self, event: tk.Event) -> None:
        self.canvas.xview_scroll(self._wheel_steps(event) * 3, "units")

    def _overflows_vertically(self) -> bool:
        first, last = self.canvas.yview()
        return not (first <= 0.0 and last >= 1.0)

    # ---- layout ----------------------------------------------------------- #
    def _on_configure(self, event: tk.Event) -> None:
        width, height = event.width, event.height
        last_w, last_h = self._last_size
        if abs(width - last_w) < 8 and abs(height - last_h) < 8:
            return
        self._last_size = (width, height)
        self._fit_window()
        if self._group is None:
            return
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(RESIZE_DEBOUNCE_MS, self._render)

    def _fit_window(self) -> None:
        """
        Stretch the canvas window to at least the viewport size so the inner
        frame's spacer rows/columns can centre the content, and keep the
        scrollregion in sync with whichever is larger.

        The ``update_idletasks`` is not optional.  This runs immediately after
        ``_render`` grids the cells, before Tk has recomputed the frame's
        requested size - it used to measure 265px for a group 2656px wide.
        Pinning the canvas window to that stale width then stopped the frame's
        *actual* size from ever changing again, so the ``<Configure>`` binding
        meant to correct it never fired and most of a group was unreachable at
        any scroll position.
        """
        if self._fitting:          # update_idletasks re-enters via <Configure>
            return
        self._fitting = True
        try:
            self.inner.update_idletasks()
            width = max(self.canvas.winfo_width(), self.inner.winfo_reqwidth())
            height = max(self.canvas.winfo_height(),
                         self.inner.winfo_reqheight())
            self.canvas.itemconfigure(self._window, width=width, height=height)
            self.canvas.configure(scrollregion=(0, 0, width, height))
        finally:
            self._fitting = False

    def _thumb_box(self, count: int) -> tuple[int, int, int]:
        """
        Return ``(box_w, box_h, columns)`` for a group of ``count`` images.

        A group small enough to sit in one row gets the full height of the
        panel and an equal share of its width, capped by THUMB_MAX so a lone
        landscape image doesn't balloon.  Once that share would squeeze the
        thumbnails below THUMB_MIN the group wraps into a contact sheet
        instead: album-sized groups are the normal case, and a single row of
        eighty images is a 15,000px strip that nobody can scan.
        """
        scale = self._scale
        canvas_w = self.canvas.winfo_width() or round(900 * scale)
        canvas_h = self.canvas.winfo_height() or round(540 * scale)
        margin, gap = round(28 * scale), round(18 * scale)

        share = (canvas_w - margin) / max(1, count) - gap
        if share >= THUMB_MIN * scale:
            reserved = round(CAPTION_SPACE * scale)
            box_h = int(max(THUMB_MIN * scale,
                            min(SOURCE_CACHE_MAX, canvas_h - reserved)))
            return int(min(THUMB_MAX * scale, share)), box_h, count

        box = int(THUMB_WRAP * scale)
        return box, box, max(1, int((canvas_w - margin) // (box + gap)))

    # ---- content ---------------------------------------------------------- #
    def _reset(self) -> None:
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
            self._resize_job = None
        # Retire any half-finished fill before the slots it holds are
        # destroyed; the generation bump makes an already-queued tick a no-op
        # even if the cancel lost a race with it.
        if self._fill_job is not None:
            self.after_cancel(self._fill_job)
            self._fill_job = None
        self._generation += 1
        self._pending = []
        for child in self.inner.winfo_children():
            child.destroy()
        self._photos.clear()
        # Weights left over from a larger group would hand the spare space to
        # a column that no longer exists and push the content off-centre.
        for index in self._weighted[0]:
            self.inner.grid_rowconfigure(index, weight=0)
        for index in self._weighted[1]:
            self.inner.grid_columnconfigure(index, weight=0)
        self._weighted = ((), ())

    def _set_spacers(self, rows: tuple[int, ...],
                     columns: tuple[int, ...]) -> None:
        for index in rows:
            self.inner.grid_rowconfigure(index, weight=1)
        for index in columns:
            self.inner.grid_columnconfigure(index, weight=1)
        self._weighted = (rows, columns)

    def show_message(self, text: str) -> None:
        self._group = None
        self._reset()
        self._set_spacers((0, 2), (0, 2))
        tk.Label(self.inner, text=text, bg=WINDOW, fg=FG_MUTED,
                 font=(FONT, 11), justify="center").grid(row=1, column=1)
        self._fit_window()

    def show_group(self, group: Sequence[ImgInfo]) -> None:
        self._group = group
        self._render()

    def clear_cache(self) -> None:
        self._cache.clear()

    def _render(self) -> None:
        """
        Lay the group out immediately, then fill the pictures in progressively.

        Decoding every image up front is what a three-image "group" could
        afford; an album of fifty cannot.  Measured on real files, decoding the
        lot synchronously blocked the UI thread for 2.3s at sixty small images
        and would comfortably clear the five seconds at which Windows paints
        "Not Responding" over the window for an album of full-size
        photographs.  So the cells - which are cheap - go down at once, and the
        thumbnails arrive a few per idle tick.

        Every slot is a fixed box, so pictures appearing later never move
        anything that is already on screen, and captions line up by
        construction rather than by measuring the tallest image first.
        """
        self._resize_job = None
        group = self._group
        if not group:
            return
        self._reset()

        box_w, box_h, columns = self._thumb_box(len(group))
        scale = self._scale
        pad = round(8 * scale)
        rows = (len(group) + columns - 1) // columns

        # Spacer rows/columns keep a small group centred in a large panel.
        self._set_spacers((0, rows + 1), (0, columns + 1))

        self._pending = []
        for position, info in enumerate(group):
            row, column = divmod(position, columns)
            cell = tk.Frame(self.inner, bg=WINDOW)
            cell.grid(row=row + 1, column=column + 1, sticky="n",
                      padx=pad, pady=round(10 * scale))
            self._pending.append((info, self._build_cell(cell, info, box_w,
                                                         box_h)))

        self._fit_window()
        self.canvas.xview_moveto(0.0)
        self.canvas.yview_moveto(0.0)

        self._generation += 1
        self._fill(0, self._generation, box_w, box_h)

    def _fill(self, start: int, generation: int, box_w: int,
              box_h: int) -> None:
        """Decode and install the next few thumbnails, then yield to Tk."""
        self._fill_job = None
        if generation != self._generation:
            return                      # a different group was selected

        for info, slot in self._pending[start:start + FILL_CHUNK]:
            if not slot.winfo_exists():
                return
            self._install(slot, info, box_w, box_h)

        following = start + FILL_CHUNK
        if following < len(self._pending):
            self._fill_job = self.after(1, self._fill, following, generation,
                                        box_w, box_h)

    def _thumbnail(self, info: ImgInfo, box_w: int,
                   box_h: int) -> ImageTk.PhotoImage | None:
        try:
            thumb = self._cache.get(info.path).copy()
            thumb.thumbnail((box_w, box_h), Image.Resampling.LANCZOS,
                            reducing_gap=2.0)
            photo = ImageTk.PhotoImage(thumb)
        except Exception:
            return None
        self._photos.append(photo)
        return photo

    def _build_cell(self, cell: tk.Frame, info: ImgInfo, box_w: int,
                    box_h: int) -> tk.Frame:
        """Everything except the picture; returns the slot to fill later."""
        scale = self._scale
        exemplar = info.index == 1
        slot = tk.Frame(cell, bg=WINDOW, width=box_w, height=box_h + 2)
        slot.pack_propagate(False)
        slot.pack()

        tk.Label(cell, text=info.new_name, bg=WINDOW,
                 fg=EXEMPLAR_FG if exemplar else FG,
                 font=(FONT_MONO, 11, "bold")).pack(pady=(round(8 * scale), 1))
        tk.Label(cell, text=os.path.basename(info.path), bg=WINDOW,
                 fg=FG_MUTED, font=(FONT, 9),
                 wraplength=max(box_w, 160), justify="center").pack()
        # Which folder the file came from.  A group here is an album, and an
        # album routinely spans every source folder that was scanned, so the
        # folder is what says where a picture actually came from.
        tk.Label(cell, text=folder_caption(info.path, box_w), bg=WINDOW,
                 fg=FG_MUTED, font=(FONT, 8),
                 wraplength=max(box_w, 160), justify="center").pack(pady=(0, 1))
        tk.Label(cell,
                 text="group exemplar" if exemplar
                      else f"{info.match:.1f}% like the exemplar",
                 bg=WINDOW, fg=EXEMPLAR_FG if exemplar else ACCENT,
                 font=(FONT_BOLD, 9)).pack(pady=(2, round(6 * scale)))
        return slot

    def _install(self, slot: tk.Frame, info: ImgInfo, box_w: int,
                 box_h: int) -> None:
        photo = self._thumbnail(info, box_w, box_h)
        if photo is None:
            tk.Label(slot, text="[cannot preview]", fg=FG_MUTED, bg=WINDOW,
                     wraplength=box_w, justify="center").place(
                relx=0.5, rely=0.5, anchor="center")
            return
        # The exemplar gets a lit border: it is the image the group was
        # measured against, and the one that becomes "<group>_1".
        holder = tk.Frame(slot, bg=EXEMPLAR_FG if info.index == 1 else BORDER,
                          padx=1, pady=1)
        holder.place(relx=0.5, rely=0.5, anchor="center")
        label = tk.Label(holder, image=photo, bg=WINDOW, bd=0, cursor="hand2")
        label.pack()
        opener = self._on_open
        if opener is not None:
            label.bind("<Double-1>", lambda _e, p=info.path: opener(p))
