"""Tkinter front-end for the image grouper."""

from __future__ import annotations

import contextlib
import os
import queue
import threading
import time
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, ttk
from typing import Any

from PIL import Image, ImageTk

from .cache import FeatureCache
from .dnd import register_drop_target
from .engine import (
    UNGROUPED_OPTIONS,
    ImgInfo,
    collect_files,
    destination_conflicts,
    group_images,
    human_size,
    is_inside,
    organise,
    scan_images,
)
from .folderpanel import FolderPanel
from .gallery import Gallery
from .resources import (
    apply_dark_titlebar,
    asset_path,
    enable_dpi_awareness,
    set_windows_app_id,
)
from .settings import MAX_THRESHOLD, MIN_THRESHOLD, Settings
from .theme import (
    ACCENT,
    BORDER,
    EDGE,
    ELEV,
    EXEMPLAR_FG,
    FG,
    FG_MUTED,
    GAP,
    GAP_SECTION,
    GAP_TIGHT,
    GAP_WIDE,
    ON_FILL,
    PAD_CARD,
    WINDOW,
    apply_theme,
)

POLL_MS = 60
REGROUP_DEBOUNCE_MS = 220


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return "<1s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def shorten(path: str, limit: int = 22) -> str:
    """
    Trim a source folder down to the part worth reading.

    Elide from the *left*: a dump's folders share a long common prefix and
    differ only at the end, so keeping the head would print the same
    unhelpful "C:\\Users\\...\\Pictures" against every single row.
    """
    if len(path) <= limit:
        return path
    parts = path.replace("/", os.sep).split(os.sep)
    tail = parts[-1]
    for count in range(2, len(parts)):
        candidate = os.sep.join(parts[-count:])
        if len(candidate) + 2 > limit:
            break
        tail = candidate
    return f"…{os.sep}{tail}"


class App(tk.Tk):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale_factor = scale
        self.settings = Settings.load()
        self.cache = FeatureCache()
        self.cache.prune()

        self.title("Image Grouper")
        self.geometry(f"{round(1240 * scale)}x{round(900 * scale)}")
        self.minsize(round(1040 * scale), round(720 * scale))

        # Tk sizes point-based fonts from this; the DPI opt-in means the OS no
        # longer stretches the window for us.
        self.tk.call("tk", "scaling", scale * 96.0 / 72.0)
        apply_theme(self, scale)
        self._set_icon()

        self.infos: list[ImgInfo] = []
        self.groups: list[list[ImgInfo]] = []
        self._path_for_item: dict[str, str] = {}
        self._group_of_item: dict[str, int] = {}
        self._shown_group: int | None = None
        self._summary = "Add a folder, then Analyse."
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._cancel: threading.Event | None = None
        self._poll_job: str | None = None
        self._regroup_job: str | None = None
        self._busy = ""                      # "", "scan", "regroup", "copy"
        self._normal_geometry = ""
        self._scan_started = 0.0
        self._phase_started = 0.0
        self._phase_total = 0
        self._phase_label = ""
        self._phase_rate = False

        self._build_ui()
        self._apply_settings()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(60, self._style_titlebar)

    # ---- construction ----------------------------------------------------- #
    def _px(self, value: float) -> int:
        return max(1, round(value * self.scale_factor))

    def _style_titlebar(self) -> None:
        apply_dark_titlebar(self, caption=WINDOW, text=FG, border=BORDER)

    def _set_icon(self) -> None:
        """Window / taskbar icon.  Missing assets must never be fatal."""
        try:
            ico = asset_path("app.ico")
            if os.path.exists(ico):
                with contextlib.suppress(tk.TclError):
                    self.iconbitmap(default=ico)      # best on Windows
            png = asset_path("app.png")
            if os.path.exists(png):
                # Tk 8.6 reads PNG natively, which keeps iconphoto happy - it
                # only accepts a tkinter PhotoImage, not a Pillow one.
                self._icon_photo = tk.PhotoImage(file=png)
                self.iconphoto(True, self._icon_photo)
        except Exception:
            pass

    def _build_ui(self) -> None:
        edge = self._px(EDGE)

        header = ttk.Frame(self, style="Window.TFrame")
        header.pack(fill="x", padx=edge, pady=(self._px(GAP_WIDE),
                                               self._px(GAP_WIDE)))
        try:
            with Image.open(asset_path("app.png")) as raw:
                logo = raw.convert("RGBA").resize(
                    (self._px(30), self._px(30)), Image.Resampling.LANCZOS)
            self._logo_photo = ImageTk.PhotoImage(logo)
            tk.Label(header, image=self._logo_photo, bg=WINDOW,
                     bd=0).pack(side="left", padx=(0, self._px(GAP_TIGHT)))
        except Exception:
            pass
        ttk.Label(header, text="Image Grouper",
                  style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="groups images into albums by what is in them"
                              "  ·  copies them out renamed, never moves or "
                              "deletes",
                  style="Tagline.TLabel").pack(side="left",
                                               padx=(self._px(GAP_WIDE), 0))

        self._build_sources(edge)
        self._build_destination(edge)
        self._build_results(edge)
        self._build_actions(edge)
        self._build_status(edge)
        self._build_context_menu()

    def _card(self, edge: int, bottom: int) -> ttk.Frame:
        """A bordered panel on the window background, with its inner padding."""
        card = ttk.Frame(self, style="Card.TFrame")
        card.pack(fill="x", padx=edge, pady=(0, self._px(bottom)))
        body = ttk.Frame(card, style="TFrame")
        body.pack(fill="both", expand=True, padx=self._px(PAD_CARD),
                  pady=self._px(PAD_CARD))
        return body

    def _rule(self, parent: tk.Misc, pady: tuple[int, int]) -> None:
        tk.Frame(parent, background=BORDER, height=1).pack(fill="x", pady=pady)

    def _build_sources(self, edge: int) -> None:
        self.recursive = tk.BooleanVar(value=True)
        self.use_cache = tk.BooleanVar(value=True)
        self.threshold = tk.IntVar(value=self.settings.threshold)

        body = self._card(edge, GAP_SECTION)

        legend = ttk.Frame(body, style="TFrame")
        legend.pack(fill="x", pady=(0, self._px(GAP)))
        ttk.Label(legend, text="SOURCE FOLDERS",
                  style="Legend.TLabel").pack(side="left")
        self.folder_count = ttk.Label(legend, text="", style="Muted.TLabel")
        self.folder_count.pack(side="right")

        self.folder_panel = FolderPanel(
            body, self.scale_factor, on_open=self._open_path,
            on_change=self._on_folders_changed,
            dialog_title="Add a folder of images",
            empty_hint="Drop folders of images here")
        self.folder_panel.pack(fill="x")

        self._rule(body, (self._px(PAD_CARD), self._px(GAP_WIDE)))

        options = ttk.Frame(body, style="TFrame")
        options.pack(fill="x")

        # Analyse is the card's one primary action, so it is packed first and
        # sits hard right; everything else on this row is a modifier.
        self.scan_btn = ttk.Button(options, text="Analyse", width=11,
                                   command=self._on_analyse_clicked)
        self.scan_btn.pack(side="right")

        threshold = ttk.Frame(options, style="TFrame")
        threshold.pack(side="right", padx=(self._px(GAP_WIDE),
                                           self._px(GAP_WIDE)))
        ttk.Label(threshold, text="Similarity", style="Muted.TLabel").pack(
            side="left", padx=(0, self._px(GAP)))
        self.thr_scale = ttk.Scale(threshold, from_=MIN_THRESHOLD,
                                   to=MAX_THRESHOLD, command=self._on_threshold,
                                   length=self._px(190))
        self.thr_scale.pack(side="left")
        self.thr_label = ttk.Label(threshold, text="75%", width=5,
                                   anchor="e", style="Value.TLabel")
        self.thr_label.pack(side="left", padx=(self._px(GAP), 0))

        ttk.Checkbutton(options, text="Include subfolders",
                        variable=self.recursive).pack(
            side="left", padx=(0, self._px(GAP_WIDE)))
        ttk.Checkbutton(options, text="Reuse cached features",
                        variable=self.use_cache).pack(side="left")
        # No fixed width: the label grows a digit at a time with the cache
        # count, and a fixed one clipped "Clear cache (12,345)".
        self.cache_btn = ttk.Button(options, text="Clear cache",
                                    style="Ghost.TButton",
                                    command=self._clear_cache)
        self.cache_btn.pack(side="left", padx=(self._px(GAP_WIDE), 0))

        # Only now - Scale.set() fires the command, which needs thr_label.
        self.thr_scale.set(self.threshold.get())
        # Regroup on release rather than on every pixel of drag: the grouping
        # is fast but not free, and mid-drag results are never looked at.
        self.thr_scale.bind("<ButtonRelease-1>", self._schedule_regroup)
        self._on_folders_changed()

    def _build_destination(self, edge: int) -> None:
        self.destination = tk.StringVar(value="")
        self.subfolders = tk.BooleanVar(value=False)
        self.ungrouped = tk.StringVar(value=UNGROUPED_OPTIONS[0])

        body = self._card(edge, GAP_SECTION)

        legend = ttk.Frame(body, style="TFrame")
        legend.pack(fill="x", pady=(0, self._px(GAP)))
        ttk.Label(legend, text="DESTINATION",
                  style="Legend.TLabel").pack(side="left")
        self.dest_hint = ttk.Label(legend, text="", style="Muted.TLabel")
        self.dest_hint.pack(side="right")

        row = ttk.Frame(body, style="TFrame")
        row.pack(fill="x")
        ttk.Entry(row, textvariable=self.destination).pack(
            side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse…", width=12, style="Secondary.TButton",
                   command=self._pick_destination).pack(
            side="left", padx=(self._px(GAP), 0))

        options = ttk.Frame(body, style="TFrame")
        options.pack(fill="x", pady=(self._px(GAP_WIDE), 0))
        ttk.Checkbutton(options, text="A subfolder per group",
                        variable=self.subfolders).pack(side="left")
        ttk.Label(options, text="Images that match nothing:",
                  style="Muted.TLabel").pack(
            side="left", padx=(self._px(GAP_WIDE), self._px(GAP)))
        combo = ttk.Combobox(options, textvariable=self.ungrouped,
                             values=list(UNGROUPED_OPTIONS), state="readonly",
                             width=24)
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", self._schedule_regroup)

        # A drop anywhere on this card sets the destination, which is why the
        # card is armed rather than just the entry: the entry is a thin strip
        # and the whole panel is what reads as "the destination".
        if register_drop_target(body, self._on_destination_dropped):
            self.dest_hint.config(text="or drop a folder here")

    def _build_results(self, edge: int) -> None:
        self.paned = ttk.Panedwindow(self, orient="horizontal")
        self.paned.pack(fill="both", expand=True, padx=edge,
                        pady=(0, self._px(GAP_SECTION)))

        left = ttk.Frame(self.paned, style="Card.TFrame")
        header = ttk.Frame(left, style="TFrame")
        header.grid(row=0, column=0, columnspan=2, sticky="ew",
                    padx=self._px(PAD_CARD),
                    pady=(self._px(PAD_CARD), self._px(GAP)))
        ttk.Label(header, text="GROUPS", style="Legend.TLabel").pack(
            side="left")
        self.group_count = ttk.Label(header, text="", style="Muted.TLabel")
        self.group_count.pack(side="right")

        columns = ("original", "match", "folder")
        self.tree = ttk.Treeview(left, columns=columns, show="tree headings",
                                 selectmode="extended")
        # Headings follow their column's own alignment; a centred heading over
        # left-aligned data reads as a different column to the data.
        self.tree.heading("#0", text="New name", anchor="w")
        self.tree.heading("original", text="Original name", anchor="w")
        self.tree.heading("match", text="Match", anchor="center")
        self.tree.heading("folder", text="From", anchor="w")
        self.tree.column("#0", width=self._px(150), minwidth=self._px(120),
                         stretch=False)
        self.tree.column("original", width=self._px(190),
                         minwidth=self._px(110), stretch=True)
        self.tree.column("match", width=self._px(66), stretch=False,
                         anchor="center")
        self.tree.column("folder", width=self._px(170), stretch=False)
        self.tree.tag_configure("group", foreground=ACCENT)
        self.tree.tag_configure("exemplar", foreground=EXEMPLAR_FG)

        ysb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=ysb.set)
        self.tree.grid(row=1, column=0, sticky="nsew")
        ysb.grid(row=1, column=1, sticky="ns")
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        # An empty Treeview draws nothing at all, so on a first run the widest
        # column of the window said less than the preview beside it.  This
        # label covers the rows only while there are none.
        self.tree_empty = ttk.Label(
            self.tree, style="Muted.TLabel", anchor="center", justify="center",
            text="Albums appear here, with the name each\n"
                 "copy will be given.\n\n"
                 "Add source folders, then Analyse.")
        self._sync_tree_empty()

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.bind("<Double-1>", self._on_tree_double)
        self.tree.bind("<Button-3>", self._on_tree_right_click)
        self.paned.add(left, weight=3)

        self.gallery = Gallery(self.paned, self.scale_factor,
                               on_open=self._open_path)
        self.paned.add(self.gallery, weight=5)
        self.after(120, self._init_sash)

    def _build_actions(self, edge: int) -> None:
        body = self._card(edge, GAP_SECTION)
        row = ttk.Frame(body, style="TFrame")
        row.pack(fill="x")

        # Copying is the app's whole point and it only ever adds files, so it
        # is the one filled green and sits at the end of the row.
        self.copy_btn = ttk.Button(
            row, text="Copy to destination", style="Go.TButton",
            state="disabled", command=self._on_copy_clicked)
        self.copy_btn.pack(side="right")

        ttk.Button(row, text="Expand all", style="Ghost.TButton",
                   command=lambda: self._set_open(True)).pack(side="left")
        ttk.Button(row, text="Collapse all", style="Ghost.TButton",
                   command=lambda: self._set_open(False)).pack(
            side="left", padx=(self._px(GAP_TIGHT), 0))
        self.plan_label = ttk.Label(row, text="", style="Muted.TLabel")
        self.plan_label.pack(side="left", padx=(self._px(GAP_WIDE), 0))

    def _build_status(self, edge: int) -> None:
        tk.Frame(self, background=BORDER, height=1).pack(fill="x")
        bar = ttk.Frame(self, style="Window.TFrame")
        bar.pack(fill="x", padx=edge, pady=self._px(GAP_WIDE))
        self.status = tk.StringVar(value=self._summary)
        ttk.Label(bar, textvariable=self.status, anchor="w",
                  style="WindowMuted.TLabel").pack(side="left", fill="x",
                                                   expand=True)
        self.progress = ttk.Progressbar(bar, mode="determinate",
                                        length=self._px(240))
        self.progress.pack(side="right", padx=(self._px(GAP_WIDE), 0))

    def _build_context_menu(self) -> None:
        self.menu = tk.Menu(self, tearoff=0, bg=ELEV, fg=FG,
                            activebackground=ACCENT, activeforeground=ON_FILL,
                            activeborderwidth=0, bd=0,
                            disabledforeground=FG_MUTED)
        self.menu.add_command(label="Open", command=self._open_selected)
        self.menu.add_command(label="Show in Explorer",
                              command=self._reveal_selected)
        self.menu.add_command(label="Copy full path",
                              command=self._copy_selected_path)

    def _init_sash(self) -> None:
        # Left alone, the tree's requested width decides the split and the
        # preview starts squashed - but splitting purely by ratio clips the
        # From column.  Show the whole tree unless that would eat the panel.
        total = self.paned.winfo_width()
        if total <= 200:
            return
        saved = self.settings.sash
        if 120 < saved < total - 200:
            self.paned.sashpos(0, saved)
            return
        columns = self._px(150 + 190 + 66 + 170 + 22)
        self.paned.sashpos(0, int(min(columns, total * 0.55)))

    # ---- settings --------------------------------------------------------- #
    def _apply_settings(self) -> None:
        # The source list is intentionally not restored - every launch starts
        # with nothing queued up to scan.  The destination is, because it is
        # nearly always the same folder run after run.
        saved = self.settings
        self.recursive.set(saved.recursive)
        self.use_cache.set(saved.use_cache)
        self.thr_scale.set(saved.threshold)
        self.destination.set(saved.destination)
        self.subfolders.set(saved.subfolders)
        if saved.ungrouped in UNGROUPED_OPTIONS:
            self.ungrouped.set(saved.ungrouped)
        if saved.geometry:
            self._normal_geometry = saved.geometry
            with contextlib.suppress(tk.TclError):
                self.geometry(saved.geometry)
        # Set the geometry first, so un-maximising lands on the saved size.
        if saved.maximized:
            with contextlib.suppress(tk.TclError):
                self.state("zoomed")
        self.bind("<Configure>", self._on_window_configure)
        self._refresh_cache_label()

    def _on_window_configure(self, event: tk.Event) -> None:
        """
        Track the window's *restored* geometry.

        While maximised, ``winfo_geometry()`` reports the maximised size, so
        saving it blindly reopened the app at full-screen size but in normal
        state - the right size with no way to un-maximise back.
        """
        if event.widget is not self:
            return                            # child widgets configure too
        with contextlib.suppress(tk.TclError):
            if self.state() == "normal":
                self._normal_geometry = self.winfo_geometry()

    def _capture_settings(self) -> Settings:
        sash = 0
        with contextlib.suppress(tk.TclError):
            sash = self.paned.sashpos(0)
        maximized = False
        with contextlib.suppress(tk.TclError):
            maximized = self.state() == "zoomed"
        return Settings(
            recursive=bool(self.recursive.get()),
            threshold=int(self.threshold.get()),
            use_cache=bool(self.use_cache.get()),
            destination=self.destination.get().strip(),
            subfolders=bool(self.subfolders.get()),
            ungrouped=self.ungrouped.get(),
            geometry=self._normal_geometry or self.winfo_geometry(),
            maximized=maximized,
            sash=sash,
        )

    def _refresh_cache_label(self) -> None:
        count = self.cache.count()
        self.cache_btn.config(
            text="Clear cache" if not count else f"Clear cache ({count:,})",
            state="normal" if count else "disabled")

    def _clear_cache(self) -> None:
        if messagebox.askyesno(
                "Clear cache",
                "Discard all stored image features?\n\nThe next analysis will "
                "have to read every file again."):
            self.cache.clear()
            self._refresh_cache_label()

    # ---- folders ---------------------------------------------------------- #
    def _folders(self) -> list[str]:
        return self.folder_panel.folders()

    def _sync_tree_empty(self) -> None:
        groups = len(self.tree.get_children())
        if groups:
            self.tree_empty.place_forget()
        else:
            self.tree_empty.place(relx=0.5, rely=0.42, anchor="center")
        self.group_count.config(
            text="" if not groups else
            f"{groups} group" if groups == 1 else f"{groups} groups")

    def _on_folders_changed(self) -> None:
        count = len(self._folders())
        self.folder_count.config(
            text="" if not count else
            f"{count} folder" if count == 1 else f"{count} folders")

    def _pick_destination(self) -> None:
        current = self.destination.get().strip()
        chosen = filedialog.askdirectory(
            title="Where should the organised copies go?",
            initialdir=current or None)
        if chosen:
            self.destination.set(os.path.normpath(chosen))

    def _on_destination_dropped(self, folders: list[str]) -> None:
        """Only one folder can be the destination, so take the first."""
        self.destination.set(folders[0])

    # ---- small helpers ---------------------------------------------------- #
    def _on_threshold(self, raw: str) -> None:
        value = int(round(float(raw)))
        if value != self.threshold.get():
            self.threshold.set(value)
        self.thr_label.config(text=f"{value}%")

    def _refresh_status(self) -> None:
        self.status.set(self._summary)

    def _set_open(self, opened: bool) -> None:
        for item in self.tree.get_children():
            self.tree.item(item, open=opened)

    # ---- work ------------------------------------------------------------- #
    def _set_busy(self, kind: str) -> None:
        self._busy = kind
        running = bool(kind)
        self.scan_btn.config(
            text="Cancel" if kind == "scan" else "Analyse",
            style="Danger.TButton" if kind == "scan" else "TButton",
            state="normal" if kind in ("", "scan") else "disabled")
        self.copy_btn.config(
            state="normal" if not running and self.groups else "disabled")

    def _set_phase(self, text: str, total: int, rate: bool) -> None:
        self._summary = text
        self._phase_label = text
        self._phase_started = time.monotonic()
        self._phase_total = total
        self._phase_rate = rate
        self.progress.config(maximum=max(1, total or 100), value=0)
        self._refresh_status()

    def _start(self, kind: str, target, *args) -> None:
        self._cancel = threading.Event()
        self._set_busy(kind)
        threading.Thread(target=target, args=(*args, self._cancel),
                         daemon=True).start()
        if self._poll_job is None:
            self._poll_job = self.after(POLL_MS, self._poll_queue)

    def _on_analyse_clicked(self) -> None:
        if self._busy == "scan":
            self._cancel.set()  # type: ignore[union-attr]
            self.scan_btn.config(state="disabled", text="Stopping…")
            return
        if self._busy:
            return

        folders = self._folders()
        if not folders:
            messagebox.showerror("No folders",
                                 "Add at least one folder of images.")
            return
        missing = [f for f in folders if not os.path.isdir(f)]
        if missing:
            messagebox.showerror(
                "Invalid folder",
                "These folders no longer exist:\n\n" + "\n".join(missing[:8]))
            return

        self._clear_results()
        self._scan_started = time.monotonic()
        self._set_phase("Collecting files…", 0, rate=False)
        self._start("scan", self._scan_worker, folders,
                    bool(self.recursive.get()), float(self.threshold.get()),
                    self.ungrouped.get(),
                    self.cache if self.use_cache.get() else None)

    def _scan_worker(self, folders, recursive, threshold, ungrouped, cache,
                     cancel) -> None:
        put = self._queue.put
        try:
            paths = list(collect_files(folders, recursive))
            if cancel.is_set():
                put(("cancelled", None))
                return
            put(("phase", ("Reading images…", len(paths), True)))
            infos, errors = scan_images(
                paths, progress=lambda done, _t: put(("progress", done)),
                cancel=cancel, cache=cache)
            if cancel.is_set():
                put(("cancelled", None))
                return

            put(("phase", ("Comparing and grouping…", 100, False)))
            groups = group_images(
                infos, threshold, ungrouped=ungrouped, cancel=cancel,
                progress=lambda done, _t: put(("progress", done)))
            if cancel.is_set():
                put(("cancelled", None))
                return
            put(("scanned", (infos, groups, errors)))
        except Exception:
            put(("error", traceback.format_exc()))

    def _schedule_regroup(self, _event: tk.Event | None = None) -> None:
        """Re-group from the features already in memory - no files re-read."""
        if not self.infos or self._busy:
            return
        if self._regroup_job is not None:
            self.after_cancel(self._regroup_job)
        self._regroup_job = self.after(REGROUP_DEBOUNCE_MS, self._regroup)

    def _regroup(self) -> None:
        self._regroup_job = None
        if not self.infos or self._busy:
            return
        self._set_phase("Regrouping…", 100, rate=False)
        self._start("regroup", self._regroup_worker,
                    float(self.threshold.get()), self.ungrouped.get())

    def _regroup_worker(self, threshold, ungrouped, cancel) -> None:
        put = self._queue.put
        try:
            groups = group_images(
                self.infos, threshold, ungrouped=ungrouped, cancel=cancel,
                progress=lambda done, _t: put(("progress", done)))
            if cancel.is_set():
                put(("cancelled", None))
                return
            put(("grouped", groups))
        except Exception:
            put(("error", traceback.format_exc()))

    def _poll_queue(self) -> None:
        self._poll_job = None
        latest_progress: int | None = None
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "phase":
                    text, maximum, rate = payload
                    latest_progress = None
                    self._set_phase(text, maximum, rate)
                elif kind == "progress":
                    # Coalesce: one repaint per poll, not one per file.
                    latest_progress = payload
                elif kind == "error":
                    self._finish()
                    self._summary = "Failed."
                    self._refresh_status()
                    messagebox.showerror("Something went wrong", str(payload))
                    return
                elif kind == "cancelled":
                    self._finish()
                    self.progress.config(value=0)
                    self._summary = "Cancelled."
                    self._refresh_status()
                    return
                elif kind == "scanned":
                    infos, groups, errors = payload
                    self._finish()
                    self.infos = infos
                    self._populate(groups, len(infos), errors)
                    return
                elif kind == "grouped":
                    self._finish()
                    self._populate(payload, len(self.infos), [])
                    return
                elif kind == "organised":
                    self._finish_copy(*payload)
                    return
        except queue.Empty:
            pass
        if latest_progress is not None:
            self._show_progress(latest_progress)
        self._poll_job = self.after(POLL_MS, self._poll_queue)

    def _show_progress(self, done: int) -> None:
        self.progress.config(value=done)
        if not self._phase_rate or not self._phase_total:
            return
        elapsed = time.monotonic() - self._phase_started
        if elapsed < 0.4 or done <= 0:
            return
        rate = done / elapsed
        remaining = (self._phase_total - done) / rate if rate > 0 else 0
        self.status.set(
            f"{self._phase_label}  {done:,} / {self._phase_total:,}  ·  "
            f"{rate:,.0f} img/s  ·  {format_duration(remaining)} left")

    def _finish(self) -> None:
        self._cancel = None
        self._set_busy("")
        self._refresh_cache_label()

    def _populate(self, groups: list[list[ImgInfo]], scanned: int,
                  errors: list[tuple[str, str]]) -> None:
        self.groups = groups
        self._build_tree()

        clustered = [g for g in groups if len(g) > 1]
        in_clusters = sum(len(g) for g in clustered)
        singles = sum(1 for g in groups if len(g) == 1)
        self._summary = (
            f"{scanned:,} image(s) in "
            f"{format_duration(time.monotonic() - self._scan_started)}  ·  "
            f"{len(clustered)} group(s) covering {in_clusters:,} image(s)  ·  "
            f"{singles:,} on their own")
        if errors:
            self._summary += f"  ·  {len(errors)} skipped"
        self._refresh_status()
        self.progress.config(value=self.progress["maximum"])
        self._set_busy("")

        if groups:
            first = self.tree.get_children()
            if first:
                self.tree.selection_set(first[0])
                self.tree.focus(first[0])
        elif scanned:
            messagebox.showinfo("Nothing to organise",
                                "No images were left to group.")

    def _clear_results(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self.infos = []
        self.groups = []
        self._path_for_item.clear()
        self._group_of_item.clear()
        self._shown_group = None
        self.plan_label.config(text="")
        self._sync_tree_empty()
        self.gallery.clear_cache()
        self.gallery.show_message("Analyse a folder, then select a group to\n"
                                  "see its images side by side.")

    def _build_tree(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self._path_for_item.clear()
        self._group_of_item.clear()
        self._shown_group = None

        total = 0
        for index, group in enumerate(self.groups):
            worst = min(i.match for i in group)
            label = (f"  Group {group[0].group}   ·   {len(group)} files"
                     if len(group) > 1 else
                     f"  Group {group[0].group}   ·   on its own")
            parent = self.tree.insert(
                "", "end", text=label,
                values=("", f"{worst:.0f}%" if len(group) > 1 else "", ""),
                open=len(self.groups) <= 40, tags=("group",))
            self._group_of_item[parent] = index
            for info in group:
                item = self.tree.insert(
                    parent, "end", text="   " + info.new_name,
                    values=(os.path.basename(info.path),
                            f"{info.match:.1f}%", shorten(info.folder)),
                    tags=("exemplar",) if info.index == 1 else ())
                self._path_for_item[item] = info.path
                self._group_of_item[item] = index
                total += 1

        self._sync_tree_empty()
        self.plan_label.config(
            text=f"{total:,} file(s) will be written" if total else "")
        self.copy_btn.config(
            text=f"Copy {total:,} files  →  destination" if total
            else "Copy to destination",
            state="normal" if total and not self._busy else "disabled")

    # ---- preview & selection ---------------------------------------------- #
    def _focused_path(self) -> str | None:
        """The single file a context action should act on."""
        item = self.tree.focus() or (self.tree.selection() or (None,))[0]
        if item is None:
            return None
        path = self._path_for_item.get(item)
        if path is not None:
            return path
        children = self.tree.get_children(item)
        return self._path_for_item.get(children[0]) if children else None

    def _on_tree_select(self, _event: tk.Event | None = None) -> None:
        item = self.tree.focus() or (self.tree.selection() or (None,))[0]
        if item is None:
            return
        index = self._group_of_item.get(item)
        if index is None or index == self._shown_group:
            return
        self._shown_group = index
        self.gallery.show_group(self.groups[index])

    def _on_tree_double(self, event: tk.Event) -> str | None:
        item = self.tree.identify_row(event.y)
        path = self._path_for_item.get(item) if item else None
        if path is None:
            return None                       # group headers keep expanding
        self._open_path(path)
        return "break"

    def _on_tree_right_click(self, event: tk.Event) -> None:
        item = self.tree.identify_row(event.y)
        if not item:
            return
        if item not in self.tree.selection():
            self.tree.selection_set(item)
        self.tree.focus(item)
        state = "normal" if self._focused_path() else "disabled"
        for index in (0, 1, 2):
            self.menu.entryconfig(index, state=state)
        self.menu.tk_popup(event.x_root, event.y_root)
        self.menu.grab_release()

    # ---- shell actions ---------------------------------------------------- #
    def _open_path(self, path: str) -> None:
        from .shellops import open_file

        try:
            open_file(path)
        except Exception as exc:
            messagebox.showwarning("Cannot open", f"{path}\n\n{exc}")

    def _open_selected(self) -> None:
        path = self._focused_path()
        if path:
            self._open_path(path)

    def _reveal_selected(self) -> None:
        from .shellops import reveal_in_file_manager

        path = self._focused_path()
        if not path:
            return
        try:
            reveal_in_file_manager(path)
        except Exception as exc:
            messagebox.showwarning("Cannot show file", f"{path}\n\n{exc}")

    def _copy_selected_path(self) -> None:
        path = self._focused_path()
        if not path:
            return
        self.clipboard_clear()
        self.clipboard_append(path)
        self._summary = f"Copied path: {os.path.basename(path)}"
        self._refresh_status()

    # ---- organising ------------------------------------------------------- #
    def _on_copy_clicked(self) -> None:
        if self._busy or not self.groups:
            return
        destination = self.destination.get().strip()
        if not destination:
            messagebox.showerror("No destination",
                                 "Choose a destination folder first.")
            return
        destination = os.path.normpath(os.path.abspath(destination))

        # Writing into a scanned folder would seed the next analysis with this
        # run's own output, so the copies would be found and grouped again.
        clash = [f for f in self._folders()
                 if is_inside(destination, f)
                 or (self.recursive.get() and is_inside(f, destination))]
        if clash:
            messagebox.showerror(
                "Destination overlaps a source",
                "The destination is inside a folder being scanned (or the "
                "other way round):\n\n"
                + "\n".join(clash[:5])
                + "\n\nPick a destination outside the source folders, so the "
                  "copies are not picked up by the next analysis.")
            return

        if not os.path.isdir(destination):
            if not messagebox.askyesno(
                    "Create folder?",
                    f"{destination}\n\ndoes not exist yet.  Create it?"):
                return
            try:
                os.makedirs(destination, exist_ok=True)
            except OSError as exc:
                messagebox.showerror("Cannot create folder", str(exc))
                return

        total = sum(len(g) for g in self.groups)
        conflicts = destination_conflicts(self.groups, destination)
        if conflicts and not messagebox.askyesno(
                "Overwrite existing files?",
                f"{len(conflicts)} file(s) in the destination already have "
                f"names this run would write, for example "
                f"{', '.join(conflicts[:3])}.\n\nThey will be overwritten.  "
                f"Continue?"):
            return
        if not messagebox.askyesno(
                "Confirm",
                f"Copy {total:,} file(s) into\n{destination}?\n\n"
                f"Sources are only read - nothing is moved or deleted."):
            return

        self._set_phase("Copying…", total, rate=True)
        self._start("copy", self._copy_worker, list(self.groups), destination,
                    bool(self.subfolders.get()))

    def _copy_worker(self, groups, destination, subfolders, cancel) -> None:
        put = self._queue.put
        try:
            copied, written, errors = organise(
                groups, destination, subfolders=subfolders,
                progress=lambda done, _t: put(("progress", done)),
                cancel=cancel)
            put(("organised", (copied, written, errors, destination)))
        except Exception:
            put(("error", traceback.format_exc()))

    def _finish_copy(self, copied: int, written: int,
                     errors: list[tuple[str, str]], destination: str) -> None:
        self._finish()
        self.progress.config(value=self.progress["maximum"])
        self._summary = (f"Copied {copied:,} file(s)  ·  "
                         f"{human_size(written)}  ·  {destination}")
        self._refresh_status()
        if errors:
            messagebox.showwarning(
                "Some files could not be copied",
                f"{copied} copied, {len(errors)} failed.\n\n"
                + "\n".join(f"{os.path.basename(p)}: {e}"
                            for p, e in errors[:8]))
        else:
            messagebox.showinfo(
                "Done",
                f"{copied:,} file(s) copied into\n{destination}")

    # ---- lifecycle -------------------------------------------------------- #
    def _on_close(self) -> None:
        with contextlib.suppress(Exception):
            self._capture_settings().save()
        if self._cancel is not None:
            self._cancel.set()
        if self._poll_job is not None:
            self.after_cancel(self._poll_job)
            self._poll_job = None
        self.destroy()


def main() -> None:
    scale = enable_dpi_awareness()
    set_windows_app_id()
    App(scale).mainloop()
