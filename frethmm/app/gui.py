"""customtkinter GUI for FretHMM — lazy imports, i18n, menu bar, and lightweight result summary panel."""

from __future__ import annotations

import multiprocessing
import os
import platform
import pydoc
import queue
import shutil
import sys
import threading
import unittest
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any, Optional

import tkinter as tk
import customtkinter as ctk

try:
    from frethmm import __version__ as _VERSION
except Exception:
    _VERSION = "0.4.0"

_LOG = "log"
_PROGRESS = "progress"
_RESULT = "result"
_DONE = "done"
_ERROR = "error"
_WARNING = "warning"
_REVIEW_DONE = "review_done"

_APP_ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"


def _debug_log_path() -> Path:
    if getattr(sys, "frozen", False):
        base_dir = Path.home() / "AppData" / "Local" / "FretHMM"
    else:
        base_dir = Path(__file__).resolve().parents[2] / "logs"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "frethmm-gui.log"


def _append_debug_log(message: str) -> None:
    try:
        with _debug_log_path().open("a", encoding="utf-8") as fh:
            fh.write(message.rstrip() + "\n")
    except Exception:
        pass


def _center_on_parent(dlg: ctk.CTkToplevel, width: int, height: int) -> None:
    """Centre a Toplevel on its parent or the screen."""
    sw = dlg.winfo_screenwidth()
    sh = dlg.winfo_screenheight()
    x = (sw - width) // 2
    y = (sh - height) // 2
    dlg.geometry(f"{width}x{height}+{x}+{y}")


def _resource_path(relative: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
    return base / relative


@dataclass
class _FolderBatchJob:
    folder: str
    n_states: int
    max_iter: int
    tol: float
    workers: int
    data_mode: str
    signal_column: int
    guesses_text: str = ""
    auto_states: bool = False
    n_init: int = 10
    min_states: int = 2
    max_states: int = 6


class _Msg:
    __slots__ = ("type", "payload")

    def __init__(self, typ: str, payload: Any = None) -> None:
        self.type = typ
        self.payload = payload


def _worker(
    tasks: list[dict[str, Any]],
    cancel_event: threading.Event,
    result_queue: queue.Queue[_Msg],
) -> None:
    import pickle
    import traceback

    try:
        from frethmm.core.model import process_trace_file
    except Exception as exc:
        _append_debug_log(f"Import error in worker: {exc}\n{traceback.format_exc()}")
        result_queue.put(_Msg(_ERROR, f"Import error: {exc}\n{traceback.format_exc()}"))
        result_queue.put(_Msg(_DONE, None))
        return

    total = len(tasks)
    for i, task in enumerate(tasks):
        if cancel_event.is_set():
            result_queue.put(_Msg(_LOG, "status_cancelled"))
            break

        fp = Path(task["filepath"])
        config_bytes = task["config_bytes"]
        export_options_bytes = task["export_options_bytes"]
        output_dir = (
            Path(task["output_dir"])
            if task.get("output_dir") is not None
            else None
        )
        result_queue.put(_Msg(_LOG, f"[{i + 1}/{total}] {fp.name}"))
        result_queue.put(_Msg(_PROGRESS, {"current": i + 1, "total": total}))

        try:
            config = pickle.loads(config_bytes)
            export_options = pickle.loads(export_options_bytes)
            r = process_trace_file(fp, config, output_dir, export_options=export_options)

            for w in r.warnings:
                result_queue.put(_Msg(_WARNING, w))

            result_queue.put(
                _Msg(
                    _RESULT,
                    {
                        "filepath": str(fp),
                        "result": r,
                        "output_dir": str(output_dir) if output_dir else None,
                    },
                )
            )
            result_queue.put(
                _Msg(
                    _LOG,
                    {
                        "key": "log_result",
                        "kwargs": {
                            "lp": f"{r.log_prob:.2f}",
                            "m": [round(m, 4) for m in r.state_means.tolist()],
                            "sig": f"{r.state_sigma:.4f}",
                        },
                    },
                )
            )
        except Exception as exc:
            tb = traceback.format_exc()
            _append_debug_log(f"Worker error for {fp}: {exc}\n{tb}")
            result_queue.put(_Msg(_ERROR, f"{exc}\n{tb}"))
            result_queue.put(
                _Msg(
                    _RESULT,
                    {"filepath": str(fp), "result": None, "error": str(exc)},
                )
            )

    result_queue.put(_Msg(_DONE, None))


def _review_worker(
    tasks: list[dict[str, Any]],
    output_name: str,
    rows: int,
    cols: int,
    cancel_event: threading.Event,
    result_queue: queue.Queue[_Msg],
) -> None:
    import pickle
    import traceback

    try:
        from frethmm.core.model import process_trace_file
        from frethmm.domain.models import ExportOptions
        from frethmm.viz.review_grid import plot_review_grid
    except Exception as exc:
        _append_debug_log(f"Import error in review worker: {exc}\n{traceback.format_exc()}")
        result_queue.put(_Msg(_ERROR, f"Import error: {exc}\n{traceback.format_exc()}"))
        result_queue.put(_Msg(_DONE, None))
        return

    try:
        export_options = ExportOptions.classified_only()
        results = []
        total = len(tasks)
        first_output_dir = tasks[0].get("output_dir")
        first_filepath = Path(tasks[0]["filepath"])
        out_dir = Path(first_output_dir) if first_output_dir else first_filepath.parent

        for i, task in enumerate(tasks):
            if cancel_event.is_set():
                result_queue.put(_Msg(_LOG, "status_cancelled"))
                result_queue.put(_Msg(_DONE, None))
                return
            filepath = Path(task["filepath"])
            config = pickle.loads(task["config_bytes"])
            result_output_dir = (
                Path(task["output_dir"])
                if task.get("output_dir") is not None
                else None
            )
            result_queue.put(_Msg(_LOG, f"[{i + 1}/{total}] {filepath.name}"))
            result_queue.put(_Msg(_PROGRESS, {"current": i + 1, "total": total}))
            result = process_trace_file(
                filepath,
                config,
                result_output_dir,
                export_options=export_options,
            )
            results.append(result)
            for warning in result.warnings:
                result_queue.put(_Msg(_WARNING, warning))

        output_path = out_dir / output_name
        image_paths = plot_review_grid(
            results,
            config,
            output_path,
            rows=rows,
            cols=cols,
        )
        result_queue.put(
            _Msg(
                _REVIEW_DONE,
                {
                    "image_paths": [str(path) for path in image_paths],
                    "count": len(results),
                },
            )
        )
    except Exception as exc:
        tb = traceback.format_exc()
        _append_debug_log(f"Review worker error: {exc}\n{tb}")
        result_queue.put(_Msg(_ERROR, f"{exc}\n{tb}"))
    finally:
        result_queue.put(_Msg(_DONE, None))


def _detect_fonts() -> dict[str, tuple[str, int, str]]:
    system = platform.system()
    if system == "Windows":
        family = "Segoe UI"
        mono = "Consolas"
    elif system == "Darwin":
        family = "Helvetica Neue"
        mono = "Menlo"
    else:
        family = "Helvetica"
        mono = "DejaVu Sans Mono"

    return {
        "title": (family, 16, "bold"),
        "heading": (family, 11, "bold"),
        "label": (family, 10),
        "button": (family, 10),
        "entry": (family, 10),
        "tree": (family, 9),
        "tree_heading": (family, 9, "bold"),
        "log": (mono, 10),
        "status": (family, 9),
    }


class _App:
    def __init__(self, root: ctk.CTk, fonts: dict) -> None:
        self.root = root
        self.fonts = fonts
        self.selected_files: list[str] = []
        self.folder_jobs: list[_FolderBatchJob] = []
        self.output_dir: Optional[str] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()
        self._result_queue: queue.Queue[_Msg] = queue.Queue()
        self._after_id: Optional[str] = None
        self._lang = "en"
        self._status_key = "status_ready"
        self._status_kwargs: dict[str, Any] = {}
        self._progress_text = "0/0"
        self._result_stats = {"ok": 0, "warnings": 0, "errors": 0}
        self._classified_outputs: dict[str, Path] = {}
        self._results_map: dict[str, Any] = {}
        self._selected_result_path: Optional[str] = None
        self._last_output_path: Optional[str] = None
        self._tree: Optional[ttk.Treeview] = None
        self._log_text: Optional[tk.Text] = None
        self._export_classified_var = tk.BooleanVar(value=True)
        self._export_summary_var = tk.BooleanVar(value=False)
        self._export_report_var = tk.BooleanVar(value=False)
        self._export_path_var = tk.BooleanVar(value=False)
        self._export_dwell_var = tk.BooleanVar(value=False)
        self._low_state_tail_trim_var = tk.StringVar(value="")
        self._auto_states_var = tk.BooleanVar(value=False)
        self._n_init_var = tk.IntVar(value=10)
        self._min_states_var = tk.IntVar(value=2)
        self._max_states_var = tk.IntVar(value=6)
        self._review_rows_var = tk.IntVar(value=4)
        self._review_cols_var = tk.IntVar(value=8)
        self._review_output_var = tk.StringVar(value="review_grid.png")
        self._review_image_paths: list[str] = []
        self._events_output_dir: Optional[str] = None

    def _t(self, key: str, **kwargs) -> str:
        from frethmm.app.i18n import t, get_language

        lang = get_language()
        if lang != self._lang:
            self._lang = lang
        return t(key, **kwargs)

    def build(self) -> None:
        self._build_menu()

        # Layout Split: Left (Flat Controls), Right (Hidden Runtime Summary)
        self._main_layout = ctk.CTkFrame(self.root, fg_color="transparent")
        self._main_layout.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self._left_container = ctk.CTkFrame(self._main_layout, fg_color="transparent")
        self._left_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=0)

        left_panel = ctk.CTkFrame(self._left_container, fg_color="transparent")
        left_panel.pack(fill=tk.BOTH, expand=True)

        self._right_frame = ctk.CTkFrame(self._main_layout, corner_radius=10, width=220)
        self._right_frame.pack_propagate(False)
        self._runtime_panel_visible = False

        # Header Frame in left scroll
        title_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        title_frame.pack(fill=tk.X, pady=(0, 10))
        self._title_label = ctk.CTkLabel(
            title_frame,
            text="FretHMM",
            font=ctk.CTkFont(family=self.fonts["title"][0], size=22, weight="bold"),
            text_color="#1565C0",
        )
        self._title_label.pack(side=tk.LEFT)
        self._subtitle_label = ctk.CTkLabel(
            title_frame,
            text=f"  {self._t('subtitle')}  v{_VERSION}",
            font=ctk.CTkFont(family=self.fonts["label"][0], size=12),
            text_color="#757575",
        )
        self._subtitle_label.pack(side=tk.LEFT, padx=(8, 0), pady=(8, 0))

        # Dynamic Theme Toggle
        self._theme_btn = ctk.CTkButton(
            title_frame,
            text="🌓 Theme",
            width=60,
            height=26,
            command=self._toggle_theme,
        )
        self._theme_btn.pack(side=tk.RIGHT)
        self._runtime_toggle_btn = ctk.CTkButton(
            title_frame,
            text="Show Runtime",
            width=100,
            height=26,
            command=self._toggle_runtime_panel,
        )
        self._runtime_toggle_btn.pack(side=tk.RIGHT, padx=(0, 8))

        self._build_files_section(left_panel)

        params_output_row = ctk.CTkFrame(left_panel, fg_color="transparent")
        params_output_row.pack(fill=tk.X, pady=(0, 8))
        params_output_row.grid_columnconfigure(0, weight=3)
        params_output_row.grid_columnconfigure(1, weight=2)

        self._build_params_section(params_output_row, column=0, padx=(0, 4))
        self._build_output_section(params_output_row, column=1, padx=(4, 0))

        action_review_row = ctk.CTkFrame(left_panel, fg_color="transparent")
        action_review_row.pack(fill=tk.X, pady=(0, 8))
        action_review_row.grid_columnconfigure(0, weight=3)
        action_review_row.grid_columnconfigure(1, weight=2)

        self._build_review_grid_section(action_review_row, column=0, padx=(0, 4))
        self._build_action_section(action_review_row, column=1, padx=(4, 0))
        self._build_post_section(left_panel)
        self._build_status_bar(left_panel)
        
        # Build Visualization Canvas
        self._build_result_summary_panel(self._right_frame)
        self._hide_runtime_panel()
        self._update_mode_controls()
        self._on_auto_states_changed()

        # Update styling colors for the first time
        self._update_component_colors()

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(
            label=self._t("menu_file_add"), command=self._add_files
        )
        file_menu.add_command(
            label=self._t("menu_file_add_folder"), command=self._add_folder
        )
        file_menu.add_command(
            label=self._t("menu_file_clear"), command=self._clear_files
        )
        file_menu.add_separator()
        file_menu.add_command(
            label=self._t("menu_file_exit"), command=self.root.quit
        )
        menubar.add_cascade(label=self._t("menu_file"), menu=file_menu)
        self._file_menu = file_menu

        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(
            label=self._t("menu_settings_params"),
            command=self._show_params_dialog,
        )

        lang_menu = tk.Menu(settings_menu, tearoff=0)
        lang_menu.add_command(
            label=self._t("menu_settings_lang_en"),
            command=lambda: self._switch_language("en"),
        )
        lang_menu.add_command(
            label=self._t("menu_settings_lang_zh"),
            command=lambda: self._switch_language("zh"),
        )
        settings_menu.add_cascade(
            label=self._t("menu_settings_lang"), menu=lang_menu
        )

        theme_menu = tk.Menu(settings_menu, tearoff=0)
        theme_menu.add_command(
            label=self._t("menu_settings_theme_light"),
            command=lambda: self._set_theme("Light"),
        )
        theme_menu.add_command(
            label=self._t("menu_settings_theme_dark"),
            command=lambda: self._set_theme("Dark"),
        )
        theme_menu.add_command(
            label=self._t("menu_settings_theme_system"),
            command=lambda: self._set_theme("System"),
        )
        settings_menu.add_cascade(
            label=self._t("menu_settings_theme"), menu=theme_menu
        )

        menubar.add_cascade(label=self._t("menu_settings"), menu=settings_menu)
        self._settings_menu = settings_menu
        self._lang_menu = lang_menu
        self._theme_menu = theme_menu

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(
            label=self._t("menu_help_about"), command=self._show_about
        )
        menubar.add_cascade(label=self._t("menu_help"), menu=help_menu)
        self._help_menu = help_menu

        self.root.config(menu=menubar)
        self._menubar = menubar

    def _build_files_section(self, parent: ctk.CTkFrame) -> None:
        self._file_frame = ctk.CTkFrame(parent)
        self._file_frame.pack(fill=tk.X, pady=(0, 8), padx=2)
        
        lbl = ctk.CTkLabel(self._file_frame, text=self._t('section_files'), font=ctk.CTkFont(weight="bold"))
        lbl.pack(anchor=tk.W, padx=10, pady=5)

        btn_row = ctk.CTkFrame(self._file_frame, fg_color="transparent")
        btn_row.pack(fill=tk.X, padx=10, pady=5)
        self._btn_add = ctk.CTkButton(
            btn_row, text=self._t("btn_add_files"), command=self._add_files, width=90
        )
        self._btn_add.pack(side=tk.LEFT, padx=(0, 6))
        self._btn_folder = ctk.CTkButton(
            btn_row, text=self._t("btn_add_folder"), command=self._add_folder, width=90
        )
        self._btn_folder.pack(side=tk.LEFT, padx=(0, 6))
        self._btn_clear = ctk.CTkButton(
            btn_row, text=self._t("btn_clear"), command=self._clear_files, width=80, fg_color="#757575"
        )
        self._btn_clear.pack(side=tk.LEFT)
        self._btn_remove = ctk.CTkButton(
            btn_row,
            text=self._t("btn_remove_selected"),
            command=self._delete_selected_files,
            width=110,
            fg_color="#c62828",
            hover_color="#b71c1c"
        )
        self._btn_remove.pack(side=tk.LEFT, padx=(6, 0))

        list_frame = ctk.CTkFrame(self._file_frame, fg_color="transparent")
        list_frame.pack(fill=tk.X, pady=(6, 5), padx=10)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self._file_listbox = tk.Listbox(
            list_frame,
            height=6,
            selectmode=tk.EXTENDED,
            yscrollcommand=scrollbar.set,
            font=self.fonts["entry"],
            relief=tk.FLAT,
            highlightthickness=1,
        )
        scrollbar.config(command=self._file_listbox.yview)
        self._file_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._file_listbox.bind("<Delete>", self._delete_selected_files)

    def _build_folder_jobs_section(self, parent: ctk.CTkFrame) -> None:
        self._folder_frame = ctk.CTkFrame(parent)
        self._folder_frame.pack(fill=tk.X, pady=(0, 8), padx=2)
        
        lbl = ctk.CTkLabel(self._folder_frame, text=self._t('section_folder_jobs'), font=ctk.CTkFont(weight="bold"))
        lbl.pack(anchor=tk.W, padx=10, pady=5)

        btn_row = ctk.CTkFrame(self._folder_frame, fg_color="transparent")
        btn_row.pack(fill=tk.X, padx=10, pady=5)
        self._btn_add_state_folder = ctk.CTkButton(
            btn_row,
            text=self._t("btn_add_state_folder"),
            command=self._add_state_folder_job,
            width=110
        )
        self._btn_add_state_folder.pack(side=tk.LEFT, padx=(0, 6))
        self._btn_remove_state_folder = ctk.CTkButton(
            btn_row,
            text=self._t("btn_remove_state_folder"),
            command=self._remove_selected_folder_jobs,
            width=120,
            fg_color="#c62828",
            hover_color="#b71c1c"
        )
        self._btn_remove_state_folder.pack(side=tk.LEFT, padx=(0, 6))
        self._btn_clear_state_folders = ctk.CTkButton(
            btn_row,
            text=self._t("btn_clear_state_folders"),
            command=self._clear_folder_jobs,
            width=80,
            fg_color="#757575"
        )
        self._btn_clear_state_folders.pack(side=tk.LEFT)

        # Treeview frame
        tree_frame = ctk.CTkFrame(self._folder_frame, fg_color="transparent")
        tree_frame.pack(fill=tk.X, pady=(6, 5), padx=10)
        columns = ("folder", "states", "mode", "signal_column", "files")
        self._folder_tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            height=3,
        )
        self._folder_tree.heading("folder", text=self._t("col_folder"))
        self._folder_tree.heading("states", text=self._t("col_states"))
        self._folder_tree.heading("mode", text=self._t("col_mode"))
        self._folder_tree.heading("signal_column", text=self._t("col_signal_column"))
        self._folder_tree.heading("files", text=self._t("col_files"))
        self._folder_tree.column("folder", width=200, minwidth=100)
        self._folder_tree.column("states", width=60, minwidth=50, anchor=tk.CENTER)
        self._folder_tree.column("mode", width=100, minwidth=80, anchor=tk.CENTER)
        self._folder_tree.column(
            "signal_column", width=80, minwidth=60, anchor=tk.CENTER
        )
        self._folder_tree.column("files", width=60, minwidth=50, anchor=tk.CENTER)
        folder_scroll = ttk.Scrollbar(
            tree_frame, orient=tk.VERTICAL, command=self._folder_tree.yview
        )
        self._folder_tree.configure(yscrollcommand=folder_scroll.set)
        self._folder_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        folder_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_params_section(
        self,
        parent: ctk.CTkFrame,
        *,
        column: int = 0,
        padx: tuple[int, int] = (0, 0),
    ) -> None:
        self._param_frame = ctk.CTkFrame(parent)
        self._param_frame.grid(row=0, column=column, sticky="nsew", padx=padx)
        
        lbl = ctk.CTkLabel(self._param_frame, text=self._t('section_params'), font=ctk.CTkFont(weight="bold"))
        lbl.pack(anchor=tk.W, padx=10, pady=5)

        row1 = ctk.CTkFrame(self._param_frame, fg_color="transparent")
        row1.pack(fill=tk.X, pady=3, padx=10)
        self._lbl_states = ctk.CTkLabel(row1, text=self._t("label_states"))
        self._lbl_states.pack(side=tk.LEFT)

        self._states_var = tk.IntVar(value=2)
        self._states_entry = ctk.CTkEntry(row1, textvariable=self._states_var, width=50)
        self._states_entry.pack(side=tk.LEFT, padx=(4, 16))

        self._lbl_guesses = ctk.CTkLabel(row1, text=self._t("label_guesses"))
        self._lbl_guesses.pack(side=tk.LEFT)
        self._guesses_var = tk.StringVar()
        self._guesses_entry = ctk.CTkEntry(row1, textvariable=self._guesses_var, width=150)
        self._guesses_entry.pack(side=tk.LEFT, padx=(4, 0))

        # Auto-select (BIC) checkbox + min/max range, controlling whether the
        # States entry above is used or overridden by BIC model selection.
        auto_row = ctk.CTkFrame(self._param_frame, fg_color="transparent")
        auto_row.pack(fill=tk.X, pady=3, padx=10)
        self._chk_auto_states = ctk.CTkCheckBox(
            auto_row,
            text=self._t("label_auto_states"),
            variable=self._auto_states_var,
            command=self._on_auto_states_changed,
        )
        self._chk_auto_states.pack(side=tk.LEFT, padx=(0, 12))

        self._lbl_min_states = ctk.CTkLabel(auto_row, text=self._t("label_min_states"))
        self._lbl_min_states.pack(side=tk.LEFT)
        self._min_states_entry = ctk.CTkEntry(
            auto_row, textvariable=self._min_states_var, width=40
        )
        self._min_states_entry.pack(side=tk.LEFT, padx=(4, 12))

        self._lbl_max_states = ctk.CTkLabel(auto_row, text=self._t("label_max_states"))
        self._lbl_max_states.pack(side=tk.LEFT)
        self._max_states_entry = ctk.CTkEntry(
            auto_row, textvariable=self._max_states_var, width=40
        )
        self._max_states_entry.pack(side=tk.LEFT, padx=(4, 12))

        self._lbl_n_init = ctk.CTkLabel(auto_row, text=self._t("label_n_init"))
        self._lbl_n_init.pack(side=tk.LEFT)
        self._n_init_entry = ctk.CTkEntry(
            auto_row, textvariable=self._n_init_var, width=40
        )
        self._n_init_entry.pack(side=tk.LEFT, padx=(4, 0))

        row2 = ctk.CTkFrame(self._param_frame, fg_color="transparent")
        row2.pack(fill=tk.X, pady=3, padx=10)
        self._lbl_iter = ctk.CTkLabel(row2, text=self._t("label_max_iter"))
        self._lbl_iter.pack(side=tk.LEFT)
        self._iter_var = tk.IntVar(value=500)
        self._iter_entry = ctk.CTkEntry(row2, textvariable=self._iter_var, width=60)
        self._iter_entry.pack(side=tk.LEFT, padx=(4, 16))
        
        self._lbl_tol = ctk.CTkLabel(row2, text=self._t("label_tolerance"))
        self._lbl_tol.pack(side=tk.LEFT)
        self._tol_var = tk.StringVar(value="1e-4")
        self._tol_entry = ctk.CTkEntry(row2, textvariable=self._tol_var, width=70)
        self._tol_entry.pack(side=tk.LEFT, padx=(4, 16))
        
        self._lbl_workers = ctk.CTkLabel(row2, text=self._t("label_workers"))
        self._lbl_workers.pack(side=tk.LEFT)
        self._workers_var = tk.IntVar(value=1)
        self._workers_entry = ctk.CTkEntry(row2, textvariable=self._workers_var, width=50)
        self._workers_entry.pack(side=tk.LEFT, padx=(4, 0))

        row3 = ctk.CTkFrame(self._param_frame, fg_color="transparent")
        row3.pack(fill=tk.X, pady=3, padx=10)
        self._lbl_mode = ctk.CTkLabel(row3, text=self._t("label_data_mode"))
        self._lbl_mode.pack(side=tk.LEFT)
        self._mode_var = tk.StringVar(value="single_channel")
        self._mode_combo = ctk.CTkComboBox(
            row3,
            variable=self._mode_var,
            width=140,
            values=["auto", "paired_channel", "single_channel"],
            command=self._on_mode_changed,
        )
        self._mode_combo.pack(side=tk.LEFT, padx=(4, 16))
        
        self._lbl_signal_column = ctk.CTkLabel(row3, text=self._t("label_signal_column"))
        self._lbl_signal_column.pack(side=tk.LEFT)
        self._signal_column_var = tk.IntVar(value=1)
        self._signal_entry = ctk.CTkEntry(
            row3,
            textvariable=self._signal_column_var,
            width=50,
        )
        self._signal_entry.pack(side=tk.LEFT, padx=(4, 0))

    def _build_output_section(
        self,
        parent: ctk.CTkFrame,
        *,
        column: int = 0,
        padx: tuple[int, int] = (0, 0),
    ) -> None:
        self._out_frame = ctk.CTkFrame(parent)
        self._out_frame.grid(row=0, column=column, sticky="nsew", padx=padx)
        
        lbl = ctk.CTkLabel(self._out_frame, text=self._t('section_output'), font=ctk.CTkFont(weight="bold"))
        lbl.pack(anchor=tk.W, padx=10, pady=5)

        options_row = ctk.CTkFrame(self._out_frame, fg_color="transparent")
        options_row.pack(fill=tk.X, padx=10, pady=(0, 4))
        self._output_options_label = ctk.CTkLabel(
            options_row,
            text=self._t("label_output_files"),
            width=100,
            anchor=tk.W,
        )
        self._output_options_label.pack(side=tk.LEFT, padx=(0, 8))
        self._chk_export_classified = ctk.CTkCheckBox(
            options_row,
            text=self._t("output_option_classified"),
            variable=self._export_classified_var,
        )
        self._chk_export_classified.pack(side=tk.LEFT, padx=(0, 10))
        self._chk_export_classified.select()
        self._chk_export_classified.configure(state="disabled")
        self._chk_export_summary = ctk.CTkCheckBox(
            options_row,
            text=self._t("output_option_summary"),
            variable=self._export_summary_var,
        )
        self._chk_export_summary.pack(side=tk.LEFT, padx=(0, 10))
        self._chk_export_report = ctk.CTkCheckBox(
            options_row,
            text=self._t("output_option_report"),
            variable=self._export_report_var,
        )
        self._chk_export_report.pack(side=tk.LEFT, padx=(0, 10))

        options_row2 = ctk.CTkFrame(self._out_frame, fg_color="transparent")
        options_row2.pack(fill=tk.X, padx=10, pady=(0, 5))
        self._output_options_hint = ctk.CTkLabel(
            options_row2,
            text=self._t("output_options_hint"),
            text_color="#757575",
            width=100,
            anchor=tk.W,
        )
        self._output_options_hint.pack(side=tk.LEFT, padx=(0, 8))
        self._chk_export_path = ctk.CTkCheckBox(
            options_row2,
            text=self._t("output_option_path"),
            variable=self._export_path_var,
        )
        self._chk_export_path.pack(side=tk.LEFT, padx=(0, 10))
        self._chk_export_dwell = ctk.CTkCheckBox(
            options_row2,
            text=self._t("output_option_dwell"),
            variable=self._export_dwell_var,
        )
        self._chk_export_dwell.pack(side=tk.LEFT, padx=(0, 10))

        filter_row = ctk.CTkFrame(self._out_frame, fg_color="transparent")
        filter_row.pack(fill=tk.X, padx=10, pady=(0, 5))
        self._trim_filter_label = ctk.CTkLabel(
            filter_row,
            text=self._t("label_low_state_tail_trim"),
            width=100,
            anchor=tk.W,
        )
        self._trim_filter_label.pack(side=tk.LEFT, padx=(0, 8))
        self._trim_filter_entry = ctk.CTkEntry(
            filter_row,
            textvariable=self._low_state_tail_trim_var,
            width=90,
        )
        self._trim_filter_entry.pack(side=tk.LEFT, padx=(0, 8))
        self._trim_filter_hint = ctk.CTkLabel(
            filter_row,
            text=self._t("low_state_tail_trim_hint"),
            text_color="#757575",
        )
        self._trim_filter_hint.pack(side=tk.LEFT)
        
        out_row = ctk.CTkFrame(self._out_frame, fg_color="transparent")
        out_row.pack(fill=tk.X, padx=10, pady=5)
        self._btn_output = ctk.CTkButton(
            out_row, text=self._t("btn_output_folder"), command=self._select_output, width=110
        )
        self._btn_output.pack(side=tk.LEFT, padx=(0, 8))
        self._btn_output_reset = ctk.CTkButton(
            out_row,
            text=self._t("btn_output_reset"),
            command=self._reset_output,
            width=110,
            fg_color="#757575"
        )
        self._btn_output_reset.pack(side=tk.LEFT, padx=(0, 8))
        self._btn_export_classified = ctk.CTkButton(
            out_row,
            text=self._t("btn_export_classified"),
            command=self._export_selected_classified,
            width=130
        )
        self._btn_export_classified.pack(side=tk.LEFT, padx=(0, 8))
        self._output_label = ctk.CTkLabel(
            out_row,
            text=self._t("output_auto_folder"),
            text_color="#757575",
        )
        self._output_label.pack(side=tk.LEFT)

    def _build_action_section(
        self,
        parent: ctk.CTkFrame,
        *,
        column: int = 0,
        padx: tuple[int, int] = (0, 0),
    ) -> None:
        self._act_frame = ctk.CTkFrame(parent)
        self._act_frame.grid(row=0, column=column, sticky="nsew", padx=padx)

        self._run_btn = ctk.CTkButton(
            self._act_frame,
            text=self._t("btn_run"),
            command=self._run,
            font=ctk.CTkFont(weight="bold"),
            fg_color="#1565C0",
            hover_color="#0d47a1",
            width=120
        )
        self._run_btn.pack(side=tk.LEFT, padx=(0, 8))
        
        self._cancel_btn = ctk.CTkButton(
            self._act_frame,
            text=self._t("btn_cancel"),
            command=self._cancel,
            state="disabled",
            fg_color="#c62828",
            hover_color="#b71c1c",
            width=90
        )
        self._cancel_btn.pack(side=tk.LEFT, padx=(0, 16))

        self._progress_bar = ctk.CTkProgressBar(
            self._act_frame,
            width=150,
        )
        self._progress_bar.set(0)
        self._progress_bar.pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8)
        )
        self._progress_label = ctk.CTkLabel(
            self._act_frame, text="", width=60, anchor=tk.E
        )
        self._progress_label.pack(side=tk.RIGHT)

    def _build_review_grid_section(
        self,
        parent: ctk.CTkFrame,
        *,
        column: int = 0,
        padx: tuple[int, int] = (0, 0),
    ) -> None:
        self._review_frame = ctk.CTkFrame(parent)
        self._review_frame.grid(row=0, column=column, sticky="nsew", padx=padx)

        lbl = ctk.CTkLabel(
            self._review_frame,
            text=self._t("section_review_grid"),
            font=ctk.CTkFont(weight="bold"),
        )
        lbl.pack(anchor=tk.W, padx=10, pady=5)
        self._review_title_label = lbl

        row = ctk.CTkFrame(self._review_frame, fg_color="transparent")
        row.pack(fill=tk.X, padx=10, pady=(0, 6))

        self._review_rows_label = ctk.CTkLabel(row, text=self._t("label_review_rows"))
        self._review_rows_label.pack(side=tk.LEFT)
        self._review_rows_entry = ctk.CTkEntry(row, textvariable=self._review_rows_var, width=48)
        self._review_rows_entry.pack(side=tk.LEFT, padx=(4, 12))

        self._review_cols_label = ctk.CTkLabel(row, text=self._t("label_review_cols"))
        self._review_cols_label.pack(side=tk.LEFT)
        self._review_cols_entry = ctk.CTkEntry(row, textvariable=self._review_cols_var, width=48)
        self._review_cols_entry.pack(side=tk.LEFT, padx=(4, 12))

        self._review_output_label = ctk.CTkLabel(row, text=self._t("label_review_output"))
        self._review_output_label.pack(side=tk.LEFT)
        self._review_output_entry = ctk.CTkEntry(
            row,
            textvariable=self._review_output_var,
            width=180,
        )
        self._review_output_entry.pack(side=tk.LEFT, padx=(4, 12))

        self._review_btn = ctk.CTkButton(
            row,
            text=self._t("btn_review_grid"),
            command=self._run_review_grid,
            width=150,
            fg_color="#2E7D32",
            hover_color="#1B5E20",
        )
        self._review_btn.pack(side=tk.LEFT)

    def _build_results_section(self, parent: ctk.CTkFrame) -> None:
        self._results_frame = ctk.CTkFrame(parent)
        self._results_frame.pack(fill=tk.X, pady=(0, 8), padx=2)
        
        lbl = ctk.CTkLabel(self._results_frame, text=self._t('section_results'), font=ctk.CTkFont(weight="bold"))
        lbl.pack(anchor=tk.W, padx=10, pady=5)

        # Treeview frame
        tree_frame = ctk.CTkFrame(self._results_frame, fg_color="transparent")
        tree_frame.pack(fill=tk.X, pady=(6, 5), padx=10)
        columns = ("file", "states", "log_prob", "means", "sigma", "status")
        self._tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            height=5,
        )
        self._tree.heading("file", text=self._t("col_file"))
        self._tree.heading("states", text=self._t("col_states"))
        self._tree.heading("log_prob", text=self._t("col_log_prob"))
        self._tree.heading("means", text=self._t("col_means"))
        self._tree.heading("sigma", text=self._t("col_sigma"))
        self._tree.heading("status", text=self._t("col_status"))

        self._tree.column("file", width=120, minwidth=80)
        self._tree.column("states", width=50, minwidth=40, anchor=tk.CENTER)
        self._tree.column("log_prob", width=80, minwidth=60, anchor=tk.CENTER)
        self._tree.column("means", width=120, minwidth=80)
        self._tree.column("sigma", width=60, minwidth=50, anchor=tk.CENTER)
        self._tree.column("status", width=80, minwidth=50, anchor=tk.CENTER)

        tree_scroll = ttk.Scrollbar(
            tree_frame, orient=tk.VERTICAL, command=self._tree.yview
        )
        self._tree.configure(yscrollcommand=tree_scroll.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._tree.tag_configure("ok", foreground="#2E7D32")
        self._tree.tag_configure("warning", foreground="#E65100")
        self._tree.tag_configure("error", foreground="#C62828")
        self._tree.bind("<<TreeviewSelect>>", self._on_result_selected)

    def _build_log_section(self, parent: ctk.CTkFrame) -> None:
        self._log_frame = ctk.CTkFrame(parent)
        self._log_frame.pack(fill=tk.X, pady=(0, 8), padx=2)
        
        lbl = ctk.CTkLabel(self._log_frame, text=self._t('section_log'), font=ctk.CTkFont(weight="bold"))
        lbl.pack(anchor=tk.W, padx=10, pady=5)

        # Log Text
        log_frame = ctk.CTkFrame(self._log_frame, fg_color="transparent")
        log_frame.pack(fill=tk.X, pady=(6, 5), padx=10)
        self._log_text = tk.Text(
            log_frame,
            height=5,
            state=tk.DISABLED,
            font=self.fonts["log"],
            wrap=tk.WORD,
            relief=tk.FLAT,
            highlightthickness=1,
            padx=8,
            pady=6,
        )
        self._log_text.tag_configure("normal")
        self._log_text.tag_configure("warning", foreground="#E65100")
        self._log_text.tag_configure("error", foreground="#C62828")
        self._log_text.tag_configure("success", foreground="#2E7D32")
        self._log_text.tag_configure("header", foreground="#1565C0")

        log_scroll = ttk.Scrollbar(
            log_frame, orient=tk.VERTICAL, command=self._log_text.yview
        )
        self._log_text.configure(yscrollcommand=log_scroll.set)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_post_section(self, parent: ctk.CTkFrame) -> None:
        """Post-processing row with Events / Dwell-Stats / TDP buttons."""
        self._post_frame = ctk.CTkFrame(parent)
        self._post_frame.pack(fill=tk.X, pady=(4, 8), padx=2)

        lbl = ctk.CTkLabel(
            self._post_frame,
            text=self._t("section_post_processing"),
            font=ctk.CTkFont(weight="bold"),
        )
        lbl.pack(anchor=tk.W, padx=10, pady=5)
        self._post_title_label = lbl

        btn_row = ctk.CTkFrame(self._post_frame, fg_color="transparent")
        btn_row.pack(fill=tk.X, padx=10, pady=5)
        for i in range(3):
            btn_row.grid_columnconfigure(i, weight=1)

        self._btn_events = ctk.CTkButton(
            btn_row,
            text=self._t("btn_extract_events"),
            command=self._show_events_dialog,
            width=130,
            fg_color="#00838F",
            hover_color="#006064",
        )
        self._btn_events.grid(row=0, column=0, padx=(0, 6), sticky="ew")

        self._btn_dwell = ctk.CTkButton(
            btn_row,
            text=self._t("btn_dwell_stats"),
            command=self._show_dwell_dialog,
            width=130,
            fg_color="#6A1B9A",
            hover_color="#4A148C",
        )
        self._btn_dwell.grid(row=0, column=1, padx=(0, 6), sticky="ew")

        self._btn_tdp = ctk.CTkButton(
            btn_row,
            text=self._t("btn_tdp_plot"),
            command=self._show_tdp_dialog,
            width=130,
            fg_color="#E65100",
            hover_color="#BF360C",
        )
        self._btn_tdp.grid(row=0, column=2, sticky="ew")

    # -- Events dialog + runner ------------------------------------------------

    def _show_events_dialog(self) -> None:
        if not self._classified_outputs:
            messagebox.showwarning(
                self._t("dlg_events_title"),
                self._t("msg_events_no_results"),
            )
            return

        dlg = ctk.CTkToplevel(self.root)
        dlg.title(self._t("dlg_events_title"))
        dlg.geometry("440x200")
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set()
        _center_on_parent(dlg, 440, 200)

        frame = ctk.CTkFrame(dlg, corner_radius=0, fg_color="transparent")
        frame.pack(fill=tk.BOTH, expand=True, padding=20)

        ctk.CTkLabel(frame, text=self._t("dlg_events_output_dir")).grid(
            row=0, column=0, sticky=tk.W, pady=6,
        )
        out_var = tk.StringVar(value=self.output_dir or "")
        ctk.CTkEntry(frame, textvariable=out_var, width=300).grid(
            row=0, column=1, padx=(10, 0), pady=6, sticky=tk.W,
        )

        ctk.CTkLabel(frame, text=self._t("dlg_events_tail_threshold")).grid(
            row=1, column=0, sticky=tk.W, pady=6,
        )
        thresh_var = tk.StringVar(value="100.0")
        ctk.CTkEntry(frame, textvariable=thresh_var, width=80).grid(
            row=1, column=1, padx=(10, 0), pady=6, sticky=tk.W,
        )
        ctk.CTkLabel(
            frame, text=self._t("hint_events_tail_threshold"), text_color="#757575",
        ).grid(row=2, column=1, padx=(10, 0), sticky=tk.W)

        def apply():
            dlg.destroy()
            out_dir = out_var.get().strip()
            if not out_dir:
                messagebox.showerror(self._t("msg_invalid_params"), "Output directory is required.")
                return
            try:
                threshold = float(thresh_var.get().strip())
            except ValueError:
                messagebox.showerror(self._t("msg_invalid_params"), "Threshold must be a number.")
                return
            self._run_events(out_dir, threshold)

        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.grid(row=3, column=0, columnspan=2, pady=(16, 0))
        ctk.CTkButton(btn_frame, text="OK", command=apply, width=90).pack(side=tk.LEFT, padx=6)
        ctk.CTkButton(btn_frame, text="Cancel", command=dlg.destroy, width=90, fg_color="grey").pack(side=tk.LEFT, padx=6)

    def _run_events(self, output_dir: str, tail_off_threshold_seconds: float) -> None:
        import csv as csv_module

        from frethmm.core.events import (
            DETAIL_FIELDS, STAGE_SUMMARY_FIELDS, SUMMARY_FIELDS,
            event_to_detail_row, extract_events, included_statistical_events,
            overall_fields, summarize_stage_events, summarize_stage_overall,
            summarize_events, summarize_overall,
        )
        from frethmm.formats.classified_parser import read_classified_csv

        all_events = []
        per_file_summaries = []
        multistage_files = 0
        classified_paths = sorted(self._classified_outputs.values(), key=lambda p: p.name)
        for cp in classified_paths:
            try:
                sp, sm, times = read_classified_csv(cp)
                events = extract_events(sp, sm, times, cp.name, tail_off_threshold_seconds=tail_off_threshold_seconds)
            except Exception as exc:
                print(f"  ERROR {cp.name}: {exc}")
                continue
            all_events.extend(events)
            if len(sm) > 2:
                multistage_files += 1
                per_file_summaries.extend(summarize_stage_events(cp.name, events))
            else:
                per_file_summaries.append(summarize_events(cp.name, events))

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        detail_rows = [event_to_detail_row(e) for e in all_events]
        if multistage_files:
            overall_rows = summarize_stage_overall(all_events, len(classified_paths))
            binary_events = [event for event in all_events if event.stage_state_index < 0]
            if binary_events:
                binary_overall = summarize_overall(binary_events, len(classified_paths))
                binary_overall.update({"stage_state_index": "", "stage_state_mean": "", "off_state_index": ""})
                overall_rows.append(binary_overall)
        else:
            overall_rows = [summarize_overall(all_events, len(classified_paths))]

        def _w(name, fields, rows):
            p = out / name
            with p.open("w", encoding="utf-8", newline="") as h:
                w = csv_module.DictWriter(h, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)

        _w("event_details.csv", DETAIL_FIELDS, detail_rows)
        _w("event_summary.csv", STAGE_SUMMARY_FIELDS if multistage_files else SUMMARY_FIELDS, per_file_summaries)
        _w("event_stats_overall.csv", overall_fields(overall_rows[0]), overall_rows)

        self._events_output_dir = str(out)
        self._log(
            self._t("msg_events_done", n=len(classified_paths), dir=str(out)), "success"
        )
        messagebox.showinfo(
            self._t("dlg_events_title"),
            self._t("msg_events_done", n=len(classified_paths), dir=str(out)),
        )

    # -- Dwell-Stats dialog + runner -------------------------------------------

    def _show_dwell_dialog(self) -> None:
        if not self._events_output_dir:
            messagebox.showwarning(
                self._t("dlg_dwell_title"),
                self._t("msg_dwell_no_events"),
            )
            return

        dlg = ctk.CTkToplevel(self.root)
        dlg.title(self._t("dlg_dwell_title"))
        dlg.geometry("480x250")
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set()
        _center_on_parent(dlg, 480, 250)

        frame = ctk.CTkFrame(dlg, corner_radius=0, fg_color="transparent")
        frame.pack(fill=tk.BOTH, expand=True, padding=20)

        default_input = str(Path(self._events_output_dir) / "event_details.csv")
        ctk.CTkLabel(frame, text=self._t("dlg_dwell_input")).grid(
            row=0, column=0, sticky=tk.W, pady=6,
        )
        inp_entry = ctk.CTkEntry(frame, width=300)
        inp_entry.insert(0, default_input)
        inp_entry.configure(state="readonly")
        inp_entry.grid(row=0, column=1, padx=(10, 0), pady=6, sticky=tk.W)

        ctk.CTkLabel(frame, text=self._t("dlg_dwell_output_dir")).grid(
            row=1, column=0, sticky=tk.W, pady=6,
        )
        out_var = tk.StringVar(value=self._events_output_dir)
        ctk.CTkEntry(frame, textvariable=out_var, width=300).grid(
            row=1, column=1, padx=(10, 0), pady=6, sticky=tk.W,
        )

        ctk.CTkLabel(frame, text=self._t("dlg_dwell_bins")).grid(
            row=2, column=0, sticky=tk.W, pady=6,
        )
        bins_var = tk.StringVar(value="0")
        ctk.CTkEntry(frame, textvariable=bins_var, width=60).grid(
            row=2, column=1, padx=(10, 0), pady=6, sticky=tk.W,
        )

        no_fit_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            frame, text=self._t("dlg_dwell_no_fit"), variable=no_fit_var,
        ).grid(row=3, column=1, padx=(10, 0), pady=(6, 0), sticky=tk.W)

        def apply():
            dlg.destroy()
            out_dir = out_var.get().strip()
            if not out_dir:
                messagebox.showerror(self._t("msg_invalid_params"), "Output directory is required.")
                return
            try:
                bins_val = int(bins_var.get().strip())
            except ValueError:
                bins_val = 0
            self._run_dwell_stats(default_input, out_dir, None if bins_val <= 0 else bins_val, no_fit_var.get())

        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.grid(row=4, column=0, columnspan=2, pady=(16, 0))
        ctk.CTkButton(btn_frame, text="OK", command=apply, width=90).pack(side=tk.LEFT, padx=6)
        ctk.CTkButton(btn_frame, text="Cancel", command=dlg.destroy, width=90, fg_color="grey").pack(side=tk.LEFT, padx=6)

    def _run_dwell_stats(self, input_path: str, output_dir: str, bins: Optional[int], no_fit: bool) -> None:
        import csv as csv_module

        from frethmm.core.dwell_stats import (
            describe_durations, fit_dict_to_columns, fit_exponential_dwell,
            summarize_events_extended, summarize_overall_extended,
        )
        from frethmm.formats.event_details_parser import read_event_details
        from frethmm.core.events import included_statistical_events

        events = read_event_details(Path(input_path))
        if not events:
            messagebox.showerror(self._t("msg_invalid_params"), "No events found.")
            return

        included = included_statistical_events(events)
        on_durations = [e.duration_seconds for e in included if e.event_type == "ON"]
        off_durations = [e.duration_seconds for e in included if e.event_type == "OFF"]
        source_files = {e.source_file for e in events}
        multistage_output = any(event.stage_state_index >= 0 for event in events)
        stage_indices = sorted({event.stage_state_index for event in events}) if multistage_output else [-1]
        overall_rows = []
        per_file_rows = []
        for stage_index in stage_indices:
            stage_events = [event for event in events if event.stage_state_index == stage_index]
            stage_included = included_statistical_events(stage_events)
            stage_on = [event.duration_seconds for event in stage_included if event.event_type == "ON"]
            stage_off = [event.duration_seconds for event in stage_included if event.event_type == "OFF"]
            overall = summarize_overall_extended(stage_events, len(source_files))
            overall["file_count"] = len(source_files)
            overall["total_events"] = len(stage_events)
            overall["included_events"] = len(stage_included)
            if multistage_output:
                representative = stage_events[0]
                overall.update({
                    "stage_state_index": stage_index if stage_index >= 0 else "",
                    "stage_state_mean": round(representative.stage_state_mean, 6) if stage_index >= 0 else "",
                    "off_state_index": representative.off_state_index if stage_index >= 0 else "",
                })
            on_fit = fit_exponential_dwell(stage_on, n_bins=bins) if not no_fit else None
            off_fit = fit_exponential_dwell(stage_off, n_bins=bins) if not no_fit else None
            overall.update(fit_dict_to_columns(on_fit, prefix="on"))
            overall.update(fit_dict_to_columns(off_fit, prefix="off"))
            overall_rows.append(overall)
            for source in sorted(source_files):
                file_events = [event for event in stage_events if event.source_file == source]
                if not file_events:
                    continue
                row = summarize_events_extended(source, file_events)
                if multistage_output:
                    representative = file_events[0]
                    row.update({
                        "stage_state_index": stage_index if stage_index >= 0 else "",
                        "stage_state_mean": round(representative.stage_state_mean, 6) if stage_index >= 0 else "",
                        "off_state_index": representative.off_state_index if stage_index >= 0 else "",
                    })
                per_file_rows.append(row)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "dwell_stats_summary.csv").open("w", encoding="utf-8", newline="") as h:
            w = csv_module.DictWriter(h, fieldnames=list(overall_rows[0].keys()))
            w.writeheader()
            w.writerows(overall_rows)
        fns = list(per_file_rows[0].keys()) if per_file_rows else []
        with (out / "dwell_stats_per_file.csv").open("w", encoding="utf-8", newline="") as h:
            w = csv_module.DictWriter(h, fieldnames=fns)
            w.writeheader()
            w.writerows(per_file_rows)

        on_k = overall_rows[0].get("on_rate_constant", "")
        off_k = overall_rows[0].get("off_rate_constant", "")
        msg = self._t(
            "msg_dwell_done", on_n=len(on_durations), on_k=on_k,
            off_n=len(off_durations), off_k=off_k, dir=str(out),
        )
        self._log(msg, "success")
        messagebox.showinfo(self._t("dlg_dwell_title"), msg)

    # -- TDP dialog + runner ---------------------------------------------------

    def _show_tdp_dialog(self) -> None:
        dlg = ctk.CTkToplevel(self.root)
        dlg.title(self._t("dlg_tdp_title"))
        dlg.geometry("480x280")
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set()
        _center_on_parent(dlg, 480, 280)

        frame = ctk.CTkFrame(dlg, corner_radius=0, fg_color="transparent")
        frame.pack(fill=tk.BOTH, expand=True, padding=20)

        ctk.CTkLabel(frame, text=self._t("dlg_tdp_input_dir")).grid(
            row=0, column=0, sticky=tk.W, pady=6,
        )
        inp_var = tk.StringVar(value=self.output_dir or "")
        ctk.CTkEntry(frame, textvariable=inp_var, width=300).grid(
            row=0, column=1, padx=(10, 0), pady=6, sticky=tk.W,
        )

        ctk.CTkLabel(frame, text=self._t("dlg_tdp_exposure")).grid(
            row=1, column=0, sticky=tk.W, pady=6,
        )
        exp_var = tk.StringVar(value="0.1")
        ctk.CTkEntry(frame, textvariable=exp_var, width=60).grid(
            row=1, column=1, padx=(10, 0), pady=6, sticky=tk.W,
        )

        ctk.CTkLabel(frame, text=self._t("dlg_tdp_states")).grid(
            row=2, column=0, sticky=tk.W, pady=6,
        )
        states_var = tk.StringVar(value="")
        ctk.CTkEntry(frame, textvariable=states_var, width=60).grid(
            row=2, column=1, padx=(10, 0), pady=6, sticky=tk.W,
        )

        ctk.CTkLabel(frame, text=self._t("dlg_tdp_output")).grid(
            row=3, column=0, sticky=tk.W, pady=6,
        )
        out_var = tk.StringVar(value="")
        ctk.CTkEntry(frame, textvariable=out_var, width=300).grid(
            row=3, column=1, padx=(10, 0), pady=6, sticky=tk.W,
        )

        def apply():
            dlg.destroy()
            indir = Path(inp_var.get().strip())
            if not indir.is_dir():
                messagebox.showerror(self._t("msg_invalid_params"), "Invalid report directory.")
                return
            exposure = float(exp_var.get().strip())
            n_states_str = states_var.get().strip()
            n_states = int(n_states_str) if n_states_str else None
            output = out_var.get().strip() or None
            self._run_tdp(indir, exposure, n_states, output)

        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.grid(row=4, column=0, columnspan=2, pady=(16, 0))
        ctk.CTkButton(btn_frame, text="OK", command=apply, width=90).pack(side=tk.LEFT, padx=6)
        ctk.CTkButton(btn_frame, text="Cancel", command=dlg.destroy, width=90, fg_color="grey").pack(side=tk.LEFT, padx=6)

    def _run_tdp(self, input_dir: Path, exposure: float, n_states: Optional[int], output: Optional[str]) -> None:
        from frethmm.viz.tdp import generate_tdp

        report_files = sorted(
            f for f in input_dir.iterdir()
            if f.is_file() and "report" in f.name.lower() and f.suffix in (".dat", ".txt")
        )
        if not report_files:
            messagebox.showwarning(
                self._t("dlg_tdp_title"),
                self._t("msg_tdp_no_reports"),
            )
            return

        generate_tdp(input_dir, exposure=exposure, n_display_states=n_states, output=output)
        if output:
            self._log(self._t("msg_tdp_done", path=output), "success")

    def _build_status_bar(self, parent: ctk.CTkFrame) -> None:
        status_frame = ctk.CTkFrame(parent, fg_color="transparent")
        status_frame.pack(fill=tk.X, pady=(4, 0), padx=5)
        self._status_label = ctk.CTkLabel(
            status_frame, text=self._t("status_ready"), text_color="#757575"
        )
        self._status_label.pack(side=tk.LEFT)
        ctk.CTkLabel(
            status_frame, text=f"v{_VERSION}", text_color="#757575"
        ).pack(side=tk.RIGHT)

    def _build_result_summary_panel(self, parent: ctk.CTkFrame) -> None:
        self._viz_frame = ctk.CTkFrame(parent)
        self._viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        lbl = ctk.CTkLabel(
            self._viz_frame,
            text=self._t("section_runtime_panel"),
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        lbl.pack(anchor=tk.W, padx=10, pady=5)

        self._result_summary_label = lbl
        runtime_frame = ctk.CTkFrame(self._viz_frame)
        runtime_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        self._runtime_frame = runtime_frame

        self._runtime_status_label, self._runtime_status_value = self._make_summary_row(
            runtime_frame, "runtime_status", self._t(self._status_key, **self._status_kwargs)
        )
        self._runtime_progress_label, self._runtime_progress_value = self._make_summary_row(
            runtime_frame, "runtime_progress", self._progress_text
        )
        self._runtime_summary_label, self._runtime_summary_value = self._make_summary_row(
            runtime_frame, "runtime_summary",
            self._t("runtime_summary_value", ok=0, warnings=0, errors=0),
        )
        self._runtime_output_label, self._runtime_output_value = self._make_summary_row(
            runtime_frame, "runtime_last_output", self._t("result_panel_none")
        )

        selection_frame = ctk.CTkFrame(self._viz_frame)
        selection_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self._selection_frame = selection_frame
        self._selection_title_label = ctk.CTkLabel(
            selection_frame,
            text=self._t("section_result_details"),
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self._selection_title_label.pack(anchor=tk.W, padx=10, pady=(8, 4))
        self._result_file_label, self._result_file_value = self._make_summary_row(
            selection_frame, "result_file", self._t("result_panel_none")
        )
        self._result_output_label, self._result_output_value = self._make_summary_row(
            selection_frame, "result_output", self._t("result_panel_none")
        )
        self._result_metrics_label, self._result_metrics_value = self._make_summary_row(
            selection_frame, "result_metrics", self._t("result_panel_none")
        )
        self._result_warning_label, self._result_warning_value = self._make_summary_row(
            selection_frame, "result_warnings", self._t("result_panel_none")
        )
        self._set_result_summary(None)

    def _make_summary_row(
        self,
        parent: ctk.CTkFrame,
        label_key: str,
        value_text: str,
    ) -> tuple[ctk.CTkLabel, ctk.CTkLabel]:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill=tk.X, padx=10, pady=(3, 0))
        label = ctk.CTkLabel(
            row,
            text=self._t(label_key),
            width=115,
            anchor=tk.W,
        )
        label.pack(side=tk.LEFT)
        value = ctk.CTkLabel(
            row,
            text=value_text,
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=360,
        )
        value.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return label, value

    def _on_result_selected(self, _event: Optional[tk.Event] = None) -> None:
        if self._tree is None:
            return
        selected = self._tree.selection()
        if not selected:
            self._selected_result_path = None
            self._set_result_summary(None)
            return

        self._selected_result_path = selected[0]
        self._set_result_summary(selected[0])

    def _set_result_summary(self, filepath: Optional[str]) -> None:
        if filepath is None:
            self._result_file_value.configure(text=self._t("result_panel_none"))
            self._result_output_value.configure(text=self._t("result_panel_none"))
            self._result_metrics_value.configure(text=self._t("result_panel_none"))
            self._result_warning_value.configure(text=self._t("result_panel_none"))
            return

        source_path = Path(filepath)
        r = self._results_map.get(filepath)
        classified_path = self._classified_outputs.get(filepath)
        self._result_file_value.configure(text=f"{source_path.name}\n{source_path}")
        self._result_output_value.configure(
            text=str(classified_path) if classified_path is not None else self._t("result_panel_none")
        )
        if r is None:
            self._result_metrics_value.configure(text=self._t("result_panel_unfitted_short"))
            self._result_warning_value.configure(text=self._t("result_panel_none"))
            return

        self._result_metrics_value.configure(
            text=self._t(
                "result_panel_metrics",
                n=r.n_states,
                lp=f"{r.log_prob:.2f}",
                m=", ".join(f"{m:.4f}" for m in r.state_means.tolist()),
                sig=f"{r.state_sigma:.4f}",
            )
        )
        if r.warnings:
            self._result_warning_value.configure(
                text="\n".join(f"- {w}" for w in r.warnings)
            )
        else:
            self._result_warning_value.configure(text=self._t("result_panel_none"))

    def _toggle_theme(self) -> None:
        current = ctk.get_appearance_mode()
        next_mode = "Light" if current == "Dark" else "Dark"
        ctk.set_appearance_mode(next_mode)
        # Yield focus to allow customtkinter to update, then update traditional tk widgets
        self.root.after(100, self._update_component_colors)

    def _set_theme(self, mode: str) -> None:
        ctk.set_appearance_mode(mode)
        self.root.after(100, self._update_component_colors)

    def _update_component_colors(self) -> None:
        mode = ctk.get_appearance_mode()
        if mode == "Dark":
            bg = "#2b2b2b"
            fg = "#ffffff"
            select_bg = "#1f538d"
            select_fg = "#ffffff"
            tree_bg = "#2b2b2b"
            tree_fg = "#ffffff"
            tree_field = "#2b2b2b"
            tree_head_bg = "#242424"
            text_bg = "#2b2b2b"
            text_fg = "#ffffff"
        else:
            bg = "#ffffff"
            fg = "#000000"
            select_bg = "#BBDEFB"
            select_fg = "#1565C0"
            tree_bg = "#ffffff"
            tree_fg = "#000000"
            tree_field = "#ffffff"
            tree_head_bg = "#e0e0e0"
            text_bg = "#ffffff"
            text_fg = "#000000"

        # Apply listbox colors
        self._file_listbox.config(
            bg=bg, fg=fg,
            selectbackground=select_bg,
            selectforeground=select_fg,
            highlightbackground="#565b5e" if mode == "Dark" else "#bdbdbd",
            highlightcolor="#1f538d"
        )
        
        # Apply Text log colors if the log view is enabled.
        if self._log_text is not None:
            self._log_text.config(
                bg=text_bg, fg=text_fg,
                insertbackground=fg,
                highlightbackground="#565b5e" if mode == "Dark" else "#bdbdbd",
                highlightcolor="#1f538d"
            )

        # Apply Treeview styles
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Treeview", background=tree_bg, foreground=tree_fg, fieldbackground=tree_field, rowheight=28)
        style.configure("Treeview.Heading", background=tree_head_bg, foreground=fg)
        style.map("Treeview", background=[("selected", select_bg)], foreground=[("selected", select_fg)])

    def _switch_language(self, lang: str) -> None:
        from frethmm.app.i18n import set_language

        set_language(lang)
        self._lang = lang
        self._refresh_all_text()

    def _refresh_all_text(self) -> None:
        self.root.title(self._t("app_title"))
        self._subtitle_label.configure(
            text=f"  {self._t('subtitle')}  v{_VERSION}"
        )

        self._btn_add.configure(text=self._t("btn_add_files"))
        self._btn_folder.configure(text=self._t("btn_add_folder"))
        self._btn_clear.configure(text=self._t("btn_clear"))
        self._btn_remove.configure(text=self._t("btn_remove_selected"))

        self._lbl_states.configure(text=self._t("label_states"))
        self._lbl_guesses.configure(text=self._t("label_guesses"))
        self._lbl_iter.configure(text=self._t("label_max_iter"))
        self._lbl_tol.configure(text=self._t("label_tolerance"))
        self._lbl_workers.configure(text=self._t("label_workers"))
        self._lbl_mode.configure(text=self._t("label_data_mode"))
        self._lbl_signal_column.configure(text=self._t("label_signal_column"))
        self._chk_auto_states.configure(text=self._t("label_auto_states"))
        self._lbl_min_states.configure(text=self._t("label_min_states"))
        self._lbl_max_states.configure(text=self._t("label_max_states"))
        self._lbl_n_init.configure(text=self._t("label_n_init"))

        self._output_options_label.configure(text=self._t("label_output_files"))
        self._chk_export_classified.configure(text=self._t("output_option_classified"))
        self._chk_export_summary.configure(text=self._t("output_option_summary"))
        self._chk_export_report.configure(text=self._t("output_option_report"))
        self._output_options_hint.configure(text=self._t("output_options_hint"))
        self._chk_export_path.configure(text=self._t("output_option_path"))
        self._chk_export_dwell.configure(text=self._t("output_option_dwell"))
        self._trim_filter_label.configure(text=self._t("label_low_state_tail_trim"))
        self._trim_filter_hint.configure(text=self._t("low_state_tail_trim_hint"))
        self._btn_output.configure(text=self._t("btn_output_folder"))
        self._btn_output_reset.configure(text=self._t("btn_output_reset"))
        self._btn_export_classified.configure(text=self._t("btn_export_classified"))
        if not self.output_dir:
            self._output_label.configure(text=self._t("output_auto_folder"))

        self._run_btn.configure(text=self._t("btn_run"))
        self._review_title_label.configure(text=self._t("section_review_grid"))
        self._review_rows_label.configure(text=self._t("label_review_rows"))
        self._review_rows_label.configure(text=self._t("label_review_rows"))
        self._review_cols_label.configure(text=self._t("label_review_cols"))
        self._review_output_label.configure(text=self._t("label_review_output"))
        self._review_btn.configure(text=self._t("btn_review_grid"))
        self._cancel_btn.configure(text=self._t("btn_cancel"))
        self._btn_events.configure(text=self._t("btn_extract_events"))
        self._btn_dwell.configure(text=self._t("btn_dwell_stats"))
        self._btn_tdp.configure(text=self._t("btn_tdp_plot"))
        self._result_summary_label.configure(text=self._t("section_runtime_panel"))
        self._selection_title_label.configure(text=self._t("section_result_details"))
        self._runtime_status_label.configure(text=self._t("runtime_status"))
        self._runtime_progress_label.configure(text=self._t("runtime_progress"))
        self._runtime_summary_label.configure(text=self._t("runtime_summary"))
        self._runtime_output_label.configure(text=self._t("runtime_last_output"))
        self._runtime_status_value.configure(text=self._t(self._status_key, **self._status_kwargs))
        self._runtime_progress_value.configure(text=self._progress_text)
        self._runtime_summary_value.configure(
            text=self._t(
                "runtime_summary_value",
                ok=self._result_stats["ok"],
                warnings=self._result_stats["warnings"],
                errors=self._result_stats["errors"],
            )
        )
        self._runtime_output_value.configure(
            text=self._last_output_path if self._last_output_path else self._t("result_panel_none")
        )
        if self._selected_result_path:
            self._set_result_summary(self._selected_result_path)
        else:
            self._set_result_summary(None)

        if self._tree is not None:
            self._tree.heading("file", text=self._t("col_file"))
            self._tree.heading("states", text=self._t("col_states"))
            self._tree.heading("log_prob", text=self._t("col_log_prob"))
            self._tree.heading("means", text=self._t("col_means"))
            self._tree.heading("sigma", text=self._t("col_sigma"))
            self._tree.heading("status", text=self._t("col_status"))

        self._status_label.configure(text=self._t(self._status_key, **self._status_kwargs))

        self._file_menu.entryconfig(0, label=self._t("menu_file_add"))
        self._file_menu.entryconfig(1, label=self._t("menu_file_add_folder"))
        self._file_menu.entryconfig(2, label=self._t("menu_file_clear"))
        self._file_menu.entryconfig(4, label=self._t("menu_file_exit"))

        self._settings_menu.entryconfig(
            0, label=self._t("menu_settings_params")
        )
        self._settings_menu.entryconfig(
            1, label=self._t("menu_settings_lang")
        )
        self._settings_menu.entryconfig(
            2, label=self._t("menu_settings_theme")
        )
        self._lang_menu.entryconfig(0, label=self._t("menu_settings_lang_en"))
        self._lang_menu.entryconfig(1, label=self._t("menu_settings_lang_zh"))

        self._theme_menu.entryconfig(0, label=self._t("menu_settings_theme_light"))
        self._theme_menu.entryconfig(1, label=self._t("menu_settings_theme_dark"))
        self._theme_menu.entryconfig(2, label=self._t("menu_settings_theme_system"))

        self._help_menu.entryconfig(0, label=self._t("menu_help_about"))

        self._menubar.entryconfig(0, label=self._t("menu_file"))
        self._menubar.entryconfig(1, label=self._t("menu_settings"))
        self._menubar.entryconfig(2, label=self._t("menu_help"))

    def _show_params_dialog(self) -> None:
        dlg = ctk.CTkToplevel(self.root)
        dlg.title(self._t("dlg_params_title"))
        dlg.geometry("460x520")
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set()

        sw = dlg.winfo_screenwidth()
        sh = dlg.winfo_screenheight()
        x = (sw - 460) // 2
        y = (sh - 520) // 2
        dlg.geometry(f"460x520+{x}+{y}")

        frame = ctk.CTkFrame(dlg, corner_radius=0, fg_color="transparent")
        frame.pack(fill=tk.BOTH, expand=True, padding=20)

        # Grid system
        ctk.CTkLabel(frame, text=self._t("label_states")).grid(
            row=0, column=0, sticky=tk.W, pady=6
        )
        states_var = tk.IntVar(value=self._states_var.get())
        ctk.CTkEntry(
            frame, textvariable=states_var, width=80
        ).grid(row=0, column=1, padx=(10, 0), pady=6, sticky=tk.W)

        ctk.CTkLabel(frame, text=self._t("label_max_iter")).grid(
            row=1, column=0, sticky=tk.W, pady=6
        )
        iter_var = tk.IntVar(value=self._iter_var.get())
        ctk.CTkEntry(
            frame, textvariable=iter_var, width=80
        ).grid(row=1, column=1, padx=(10, 0), pady=6, sticky=tk.W)

        ctk.CTkLabel(frame, text=self._t("label_tolerance")).grid(
            row=2, column=0, sticky=tk.W, pady=6
        )
        tol_var = tk.StringVar(value=self._tol_var.get())
        ctk.CTkEntry(frame, textvariable=tol_var, width=100).grid(
            row=2, column=1, padx=(10, 0), pady=6, sticky=tk.W
        )

        ctk.CTkLabel(frame, text=self._t("label_workers")).grid(
            row=3, column=0, sticky=tk.W, pady=6
        )
        workers_var = tk.IntVar(value=self._workers_var.get())
        ctk.CTkEntry(
            frame, textvariable=workers_var, width=80
        ).grid(row=3, column=1, padx=(10, 0), pady=6, sticky=tk.W)

        ctk.CTkLabel(frame, text=self._t("label_data_mode")).grid(
            row=4, column=0, sticky=tk.W, pady=6
        )
        mode_var = tk.StringVar(value=self._mode_var.get())
        ctk.CTkComboBox(
            frame,
            variable=mode_var,
            width=140,
            values=["auto", "paired_channel", "single_channel"],
        ).grid(row=4, column=1, padx=(10, 0), pady=6, sticky=tk.W)

        ctk.CTkLabel(frame, text=self._t("label_signal_column")).grid(
            row=5, column=0, sticky=tk.W, pady=6
        )
        signal_column_var = tk.IntVar(value=self._signal_column_var.get())
        ctk.CTkEntry(frame, textvariable=signal_column_var, width=80).grid(
            row=5, column=1, padx=(10, 0), pady=6, sticky=tk.W
        )

        ctk.CTkLabel(frame, text=self._t("label_guesses")).grid(
            row=6, column=0, sticky=tk.W, pady=6
        )
        guesses_var = tk.StringVar(value=self._guesses_var.get())
        ctk.CTkEntry(frame, textvariable=guesses_var, width=220).grid(
            row=6, column=1, padx=(10, 0), pady=6, sticky=tk.W
        )

        ctk.CTkLabel(frame, text=self._t("label_n_init")).grid(
            row=7, column=0, sticky=tk.W, pady=6
        )
        n_init_var = tk.IntVar(value=self._n_init_var.get())
        ctk.CTkEntry(frame, textvariable=n_init_var, width=80).grid(
            row=7, column=1, padx=(10, 0), pady=6, sticky=tk.W
        )

        ctk.CTkLabel(frame, text=self._t("label_auto_states")).grid(
            row=8, column=0, sticky=tk.W, pady=6
        )
        auto_states_var = tk.BooleanVar(value=self._auto_states_var.get())
        ctk.CTkCheckBox(
            frame,
            text="",
            variable=auto_states_var,
        ).grid(row=8, column=1, padx=(10, 0), pady=6, sticky=tk.W)

        ctk.CTkLabel(frame, text=self._t("label_min_states")).grid(
            row=9, column=0, sticky=tk.W, pady=6
        )
        min_states_var = tk.IntVar(value=self._min_states_var.get())
        ctk.CTkEntry(frame, textvariable=min_states_var, width=60).grid(
            row=9, column=1, padx=(10, 0), pady=6, sticky=tk.W
        )

        ctk.CTkLabel(frame, text=self._t("label_max_states")).grid(
            row=10, column=0, sticky=tk.W, pady=6
        )
        max_states_var = tk.IntVar(value=self._max_states_var.get())
        ctk.CTkEntry(frame, textvariable=max_states_var, width=60).grid(
            row=10, column=1, padx=(10, 0), pady=6, sticky=tk.W
        )

        def apply():
            self._states_var.set(states_var.get())
            self._iter_var.set(iter_var.get())
            self._tol_var.set(tol_var.get())
            self._workers_var.set(workers_var.get())
            self._mode_var.set(mode_var.get())
            self._signal_column_var.set(signal_column_var.get())
            self._guesses_var.set(guesses_var.get())
            self._n_init_var.set(n_init_var.get())
            self._auto_states_var.set(auto_states_var.get())
            self._min_states_var.set(min_states_var.get())
            self._max_states_var.set(max_states_var.get())
            self._update_mode_controls()
            self._on_auto_states_changed()
            dlg.destroy()

        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.grid(row=11, column=0, columnspan=2, pady=(16, 0))
        ctk.CTkButton(btn_frame, text="OK", command=apply, width=90).pack(
            side=tk.LEFT, padx=6
        )
        ctk.CTkButton(
            btn_frame, text="Cancel", command=dlg.destroy, width=90, fg_color="grey"
        ).pack(side=tk.LEFT, padx=6)

    def _show_about(self) -> None:
        messagebox.showinfo(
            self._t("about_title"),
            self._t("about_msg", v=_VERSION),
        )

    def _show_runtime_panel(self) -> None:
        if getattr(self, "_runtime_panel_visible", False):
            return
        self._right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        self._left_container.pack_configure(padx=(0, 8))
        self._runtime_panel_visible = True
        self._runtime_toggle_btn.configure(text="Hide Runtime")

    def _hide_runtime_panel(self) -> None:
        if not getattr(self, "_runtime_panel_visible", False):
            self._right_frame.pack_forget()
            self._left_container.pack_configure(padx=0)
            self._runtime_toggle_btn.configure(text="Show Runtime")
            return
        self._right_frame.pack_forget()
        self._left_container.pack_configure(padx=0)
        self._runtime_panel_visible = False
        self._runtime_toggle_btn.configure(text="Show Runtime")

    def _toggle_runtime_panel(self) -> None:
        if getattr(self, "_runtime_panel_visible", False):
            self._hide_runtime_panel()
        else:
            self._show_runtime_panel()

    def _log(self, msg: str, tag: str = "normal") -> None:
        _append_debug_log(f"[{tag}] {msg}")
        if self._log_text is None:
            return
        self._log_text.config(state=tk.NORMAL)
        self._log_text.insert(tk.END, msg + "\n", tag)
        self._log_text.see(tk.END)
        self._log_text.config(state=tk.DISABLED)

    def _set_status(self, key: str, **kwargs) -> None:
        self._status_key = key
        self._status_kwargs = kwargs
        self._status_label.configure(text=self._t(key, **kwargs))
        self._runtime_status_value.configure(text=self._t(key, **kwargs))

    def _add_files(self) -> None:
        files = filedialog.askopenfilenames(
            title=self._t("file_dialog_title"),
            filetypes=[
                (self._t("data_files_label"), "*.dat *.txt *.csv *.tsv"),
                (self._t("all_files_label"), "*.*"),
            ],
        )
        for f in files:
            if f not in self.selected_files:
                self.selected_files.append(f)
                self._file_listbox.insert(tk.END, f)
        if files:
            self._refresh_input_status()

    def _add_folder(self) -> None:
        d = filedialog.askdirectory(
            title=self._t("folder_dialog_title")
        )
        if not d:
            return
        from frethmm.core.io import find_trace_files

        found = find_trace_files(d)
        for p in found:
            s = str(p)
            if s not in self.selected_files:
                self.selected_files.append(s)
                self._file_listbox.insert(tk.END, s)
        if not found:
            self._log(self._t("msg_no_traces"), "warning")
        else:
            self._refresh_input_status()

    def _clear_files(self) -> None:
        self.selected_files.clear()
        self.folder_jobs.clear()
        self._file_listbox.delete(0, tk.END)
        self._refresh_input_status()

    def _delete_selected_files(self, _event: tk.Event = None) -> None:
        selected = self._file_listbox.curselection()
        if not selected:
            return
        to_remove: list[str] = []
        for idx in reversed(selected):
            path = self._file_listbox.get(idx)
            to_remove.append(path)
            self._file_listbox.delete(idx)
        for p in to_remove:
            try:
                self.selected_files.remove(p)
            except ValueError:
                pass
        self._refresh_input_status()

    def _add_state_folder_job(self) -> None:
        folder = filedialog.askdirectory(title=self._t("folder_dialog_title"))
        if not folder:
            return

        from frethmm.core.io import find_trace_files

        folder_path = Path(folder)
        found = find_trace_files(folder_path)
        if not found:
            self._log(self._t("msg_no_traces"), "warning")
            return

        # Ask whether to use BIC auto-selection. If the user cancels, abort the
        # whole add flow (consistent with the legacy states prompt returning None).
        auto_states = messagebox.askyesno(
            self._t("dlg_folder_auto_states_title"),
            self._t("dlg_folder_auto_states_prompt"),
        )

        min_states = self._min_states_var.get()
        max_states = self._max_states_var.get()
        n_init = self._n_init_var.get()
        states = 0
        guesses_text = ""
        if auto_states:
            min_states = simpledialog.askinteger(
                self._t("dlg_folder_min_states_title"),
                self._t("dlg_folder_min_states_prompt"),
                parent=self.root,
                minvalue=1,
                initialvalue=min_states,
            )
            if min_states is None:
                return
            max_states = simpledialog.askinteger(
                self._t("dlg_folder_max_states_title"),
                self._t("dlg_folder_max_states_prompt"),
                parent=self.root,
                minvalue=min_states,
                initialvalue=max(max_states, min_states),
            )
            if max_states is None:
                return
        else:
            states = simpledialog.askinteger(
                self._t("dlg_folder_states_title"),
                self._t("dlg_folder_states_prompt"),
                parent=self.root,
                minvalue=1,
                initialvalue=self._states_var.get(),
            )
            if states is None:
                return
            guesses_text = self._guesses_var.get().strip()

        try:
            tol = self._parse_tol(self._tol_var.get())
            signal_column = self._signal_column_var.get()
            self._validate_signal_column(signal_column)
            if n_init < 1:
                raise ValueError(self._t("msg_invalid_n_init", v=n_init))
        except ValueError as e:
            messagebox.showerror(self._t("msg_invalid_params"), str(e))
            return

        for job in self.folder_jobs:
            if Path(job.folder) == folder_path:
                messagebox.showwarning(
                    self._t("msg_invalid_params"),
                    self._t("msg_folder_job_exists", path=str(folder_path)),
                )
                return

        job = _FolderBatchJob(
            folder=str(folder_path),
            n_states=states,
            max_iter=self._iter_var.get(),
            tol=tol,
            workers=self._workers_var.get(),
            data_mode=self._mode_var.get(),
            signal_column=signal_column,
            guesses_text=guesses_text,
            auto_states=auto_states,
            n_init=n_init,
            min_states=min_states,
            max_states=max_states,
        )
        self.folder_jobs.append(job)
        states_display = "auto" if auto_states else str(states)
        self._folder_tree.insert(
            "",
            tk.END,
            iid=str(folder_path),
            values=(
                folder_path.name,
                states_display,
                job.data_mode,
                job.signal_column,
                len(found),
            ),
        )
        self._refresh_input_status()

    def _remove_selected_folder_jobs(self) -> None:
        selected = self._folder_tree.selection()
        if not selected:
            return
        selected_set = {str(item) for item in selected}
        self.folder_jobs = [
            job for job in self.folder_jobs if job.folder not in selected_set
        ]
        for item in selected:
            self._folder_tree.delete(item)
        self._refresh_input_status()

    def _clear_folder_jobs(self) -> None:
        self.folder_jobs.clear()
        if hasattr(self, "_folder_tree"):
            for item in self._folder_tree.get_children():
                self._folder_tree.delete(item)
        self._refresh_input_status()

    def _refresh_input_status(self) -> None:
        file_count = len(self.selected_files)
        if file_count:
            self._set_status(
                "status_inputs_selected",
                files=file_count,
                folders=0,
            )
        else:
            self._set_status("status_ready")

    def _select_output(self) -> None:
        d = filedialog.askdirectory(
            title=self._t("output_dialog_title")
        )
        if d:
            self.output_dir = d
            self._output_label.configure(
                text=self._t("output_custom_folder_selected"),
                text_color="#212121",
            )
            self._set_status("status_output_selected", path=d)

    def _reset_output(self) -> None:
        self.output_dir = None
        self._output_label.configure(
            text=self._t("output_auto_folder"),
            text_color="#757575",
        )
        self._refresh_input_status()

    def _on_mode_changed(self, choice: str = "") -> None:
        self._update_mode_controls()

    def _classified_output_path(
        self,
        source_path: Path,
        output_dir: Optional[str] = None,
    ) -> Path:
        base_dir = Path(output_dir) if output_dir else self._default_output_dir_for_file(source_path)
        return base_dir / f"{source_path.stem}_classified.csv"

    @staticmethod
    def _default_output_dir_for_folder(folder_path: Path) -> Path:
        """Default GUI output directory for an input folder.

        By default the GUI writes next to the input folder, using a stable
        ``<input_folder>_output`` suffix. This keeps generated artifacts out of
        raw-data folders while preserving the custom-output-folder override.
        """
        return folder_path.parent / f"{folder_path.name}_output"

    @classmethod
    def _default_output_dir_for_file(cls, source_path: Path) -> Path:
        return cls._default_output_dir_for_folder(source_path.parent)

    def _export_selected_classified(self) -> None:
        selected_path: Optional[str] = None
        if self._tree is not None:
            selected = self._tree.selection()
            if selected:
                selected_path = selected[0]

        if selected_path is None:
            selected_path = self._selected_result_path

        if selected_path is None:
            messagebox.showwarning(
                self._t("msg_no_result_selected_title"),
                self._t("msg_no_result_selected"),
            )
            return

        source_path = Path(selected_path)
        classified_path = self._classified_outputs.get(str(source_path))
        if classified_path is None:
            classified_path = self._classified_output_path(source_path, self.output_dir)

        if not classified_path.exists():
            messagebox.showerror(
                self._t("msg_invalid_params"),
                self._t("msg_export_missing", path=str(classified_path)),
            )
            return

        target = filedialog.asksaveasfilename(
            title=self._t("btn_export_classified"),
            defaultextension=".csv",
            initialfile=classified_path.name,
            filetypes=[("CSV", "*.csv"), (self._t("all_files_label"), "*.*")],
        )
        if not target:
            return

        shutil.copyfile(classified_path, target)
        self._log(self._t("msg_export_done", path=target), "success")

    def _update_mode_controls(self) -> None:
        mode = self._mode_var.get()
        state = "normal" if mode in {"auto", "single_channel"} else "disabled"
        self._signal_entry.configure(state=state)

    def _on_auto_states_changed(self) -> None:
        """Toggle whether the fixed States entry or BIC range is active."""
        auto = self._auto_states_var.get()
        self._states_entry.configure(state="disabled" if auto else "normal")
        range_state = "normal" if auto else "disabled"
        for entry in (
            self._min_states_entry,
            self._max_states_entry,
        ):
            entry.configure(state=range_state)

    def _build_export_options(self):
        from frethmm.domain.models import ExportOptions

        return ExportOptions(
            classified_csv=True,
            summary_json=self._export_summary_var.get(),
            state_report=self._export_report_var.get(),
            state_path=self._export_path_var.get(),
            dwell_report=self._export_dwell_var.get(),
        )

    def _parse_guesses(self, guesses_text: str, n_states: int) -> Optional[list[float]]:
        guesses = None
        if guesses_text.strip():
            guesses = [float(g) for g in guesses_text.split(",")]
        if guesses is not None and len(guesses) != n_states:
            raise ValueError(self._t("msg_guess_mismatch", g=len(guesses), s=n_states))
        return guesses

    def _parse_tol(self, tol_text: str) -> float:
        try:
            return float(tol_text.strip())
        except ValueError:
            raise ValueError(self._t("msg_invalid_tol", v=tol_text.strip()))

    def _validate_signal_column(self, signal_column: int) -> None:
        if signal_column < 1:
            raise ValueError(self._t("msg_invalid_signal_column", v=signal_column))

    def _parse_low_state_tail_trim_seconds(self) -> Optional[float]:
        raw_value = self._low_state_tail_trim_var.get().strip()
        if not raw_value:
            return None
        try:
            value = float(raw_value)
        except ValueError:
            raise ValueError(self._t("msg_invalid_low_state_tail_trim", v=raw_value))
        if value <= 0:
            raise ValueError(self._t("msg_invalid_low_state_tail_trim", v=raw_value))
        return value

    def _build_config(self):
        from frethmm.domain.models import ClassificationConfig

        g_str = self._guesses_var.get().strip()
        tol_str = self._tol_var.get().strip()
        tol = self._parse_tol(tol_str)

        signal_column = self._signal_column_var.get()
        self._validate_signal_column(signal_column)
        n_init = self._n_init_var.get()
        if n_init < 1:
            raise ValueError(self._t("msg_invalid_n_init", v=n_init))
        low_state_tail_trim_seconds = self._parse_low_state_tail_trim_seconds()

        if self._auto_states_var.get():
            min_states = self._min_states_var.get()
            max_states = self._max_states_var.get()
            if min_states < 1:
                raise ValueError(self._t("msg_invalid_min_states", v=min_states))
            if max_states < min_states:
                raise ValueError(
                    self._t("msg_invalid_max_states", mn=min_states, mx=max_states)
                )
            return ClassificationConfig(
                n_states="auto",
                max_iter=self._iter_var.get(),
                tol=tol,
                guesses=None,
                workers=self._workers_var.get(),
                data_mode=self._mode_var.get(),
                signal_column=signal_column,
                low_state_tail_trim_seconds=low_state_tail_trim_seconds,
                n_init=n_init,
                min_states=min_states,
                max_states=max_states,
            )

        n_states = self._states_var.get()
        guesses = self._parse_guesses(g_str, n_states)
        return ClassificationConfig(
            n_states=n_states,
            max_iter=self._iter_var.get(),
            tol=tol,
            guesses=guesses if guesses else None,
            workers=self._workers_var.get(),
            data_mode=self._mode_var.get(),
            signal_column=signal_column,
            low_state_tail_trim_seconds=low_state_tail_trim_seconds,
            n_init=n_init,
        )

    def _build_folder_job_config(self, job: _FolderBatchJob):
        from frethmm.domain.models import ClassificationConfig

        self._validate_signal_column(job.signal_column)
        low_state_tail_trim_seconds = self._parse_low_state_tail_trim_seconds()

        if job.auto_states:
            return ClassificationConfig(
                n_states="auto",
                max_iter=job.max_iter,
                tol=job.tol,
                guesses=None,
                workers=job.workers,
                data_mode=job.data_mode,
                signal_column=job.signal_column,
                low_state_tail_trim_seconds=low_state_tail_trim_seconds,
                n_init=job.n_init,
                min_states=job.min_states,
                max_states=job.max_states,
            )

        guesses = self._parse_guesses(job.guesses_text, job.n_states)
        return ClassificationConfig(
            n_states=job.n_states,
            max_iter=job.max_iter,
            tol=job.tol,
            guesses=guesses,
            workers=job.workers,
            data_mode=job.data_mode,
            signal_column=job.signal_column,
            low_state_tail_trim_seconds=low_state_tail_trim_seconds,
            n_init=job.n_init,
        )

    def _build_tasks(self) -> list[dict[str, Any]]:
        import pickle

        tasks: list[dict[str, Any]] = []
        export_options = self._build_export_options()
        export_options_bytes = pickle.dumps(export_options)
        if self.selected_files:
            config = self._build_config()
            config_bytes = pickle.dumps(config)
            for path in self.selected_files:
                source_path = Path(path)
                task_output_dir = (
                    self.output_dir
                    if self.output_dir
                    else str(self._default_output_dir_for_file(source_path))
                )
                tasks.append(
                    {
                        "filepath": str(source_path),
                        "config_bytes": config_bytes,
                        "export_options_bytes": export_options_bytes,
                        "output_dir": task_output_dir,
                    }
                )

        if self.folder_jobs:
            from frethmm.core.io import find_trace_files

            for job in self.folder_jobs:
                config = self._build_folder_job_config(job)
                config_bytes = pickle.dumps(config)
                folder_path = Path(job.folder)
                files = find_trace_files(folder_path)
                if not files:
                    self._log(
                        self._t("msg_folder_job_empty", path=str(folder_path)),
                        "warning",
                    )
                    continue
                job_output_dir = (
                    str(Path(self.output_dir) / folder_path.name)
                    if self.output_dir
                    else str(self._default_output_dir_for_folder(folder_path))
                )
                for path in files:
                    tasks.append(
                        {
                            "filepath": str(path),
                            "config_bytes": config_bytes,
                            "export_options_bytes": export_options_bytes,
                            "output_dir": job_output_dir,
                        }
                    )
        return tasks

    def _set_ui_running(self, running: bool) -> None:
        if running:
            self._run_btn.configure(state="disabled")
            self._review_btn.configure(state="disabled")
            self._cancel_btn.configure(state="normal")
            self._set_status("status_running")
        else:
            self._run_btn.configure(state="normal")
            self._review_btn.configure(state="normal")
            self._cancel_btn.configure(state="disabled")
            self._progress_bar.set(0)
            self._progress_label.configure(text="")
            self._progress_text = "0/0"
            self._runtime_progress_value.configure(text=self._progress_text)

    def _run(self) -> None:
        if not self.selected_files and not self.folder_jobs:
            messagebox.showwarning(
                self._t("msg_no_files_title"), self._t("msg_no_files")
            )
            return

        try:
            tasks = self._build_tasks()
        except Exception as e:
            import traceback
            messagebox.showerror(
                self._t("msg_invalid_params"),
                f"{e}\n\n{traceback.format_exc()}",
            )
            return
        if not tasks:
            messagebox.showwarning(
                self._t("msg_no_files_title"), self._t("msg_no_files")
            )
            return
        self._result_stats = {"ok": 0, "warnings": 0, "errors": 0}
        self._classified_outputs = {}
        self._results_map = {}
        self._selected_result_path = None
        self._last_output_path = None
        self._runtime_summary_value.configure(
            text=self._t("runtime_summary_value", ok=0, warnings=0, errors=0)
        )
        self._runtime_output_value.configure(text=self._t("result_panel_none"))
        self._set_result_summary(None)

        if self._tree is not None:
            for item in self._tree.get_children():
                self._tree.delete(item)

        self._set_ui_running(True)
        self._cancel_event.clear()
        self._result_queue = queue.Queue()

        self._log(
            self._t(
                "log_starting_mixed",
                n=len(tasks),
                files=len(self.selected_files),
                folders=len(self.folder_jobs),
            ),
            "header",
        )

        self._worker_thread = threading.Thread(
            target=_worker,
            args=(
                tasks,
                self._cancel_event,
                self._result_queue,
            ),
            daemon=True,
        )
        self._worker_thread.start()
        self._after_id = self.root.after(80, self._poll_queue)

    def _build_review_tasks(self) -> list[dict[str, Any]]:
        import pickle

        tasks: list[dict[str, Any]] = []
        if self.selected_files:
            config = self._build_config()
            config_bytes = pickle.dumps(config)
            for path in self.selected_files:
                source_path = Path(path)
                task_output_dir = (
                    self.output_dir
                    if self.output_dir
                    else str(self._default_output_dir_for_file(source_path))
                )
                tasks.append(
                    {
                        "filepath": str(source_path),
                        "config_bytes": config_bytes,
                        "output_dir": task_output_dir,
                    }
                )

        if self.folder_jobs:
            from frethmm.core.io import find_trace_files

            for job in self.folder_jobs:
                config = self._build_folder_job_config(job)
                config_bytes = pickle.dumps(config)
                folder_path = Path(job.folder)
                files = find_trace_files(folder_path)
                if not files:
                    self._log(
                        self._t("msg_folder_job_empty", path=str(folder_path)),
                        "warning",
                    )
                    continue
                job_output_dir = (
                    str(Path(self.output_dir) / folder_path.name)
                    if self.output_dir
                    else str(self._default_output_dir_for_folder(folder_path))
                )
                for path in files:
                    tasks.append(
                        {
                            "filepath": str(path),
                            "config_bytes": config_bytes,
                            "output_dir": job_output_dir,
                        }
                    )
        return tasks

    def _run_review_grid(self) -> None:
        import traceback

        try:
            rows = int(self._review_rows_var.get())
            cols = int(self._review_cols_var.get())
            if rows < 1 or cols < 1:
                raise ValueError("rows and cols must be >= 1")
            output_name = self._review_output_var.get().strip() or self._t("review_default_prefix")
            tasks = self._build_review_tasks()
        except Exception as exc:
            messagebox.showerror(
                self._t("msg_invalid_params"),
                f"{exc}\n\n{traceback.format_exc()}",
            )
            return
        if not tasks:
            messagebox.showwarning(
                self._t("msg_no_files_title"),
                self._t("msg_review_no_files"),
            )
            return

        self._set_ui_running(True)
        self._cancel_event.clear()
        self._result_queue = queue.Queue()
        self._review_image_paths = []
        self._log(self._t("log_review_start", n=len(tasks)), "header")

        self._worker_thread = threading.Thread(
            target=_review_worker,
            args=(
                tasks,
                output_name,
                rows,
                cols,
                self._cancel_event,
                self._result_queue,
            ),
            daemon=True,
        )
        self._worker_thread.start()
        self._after_id = self.root.after(80, self._poll_queue)

    def _cancel(self) -> None:
        self._cancel_event.set()
        self._log(self._t("log_cancelling"), "warning")
        self._set_status("status_cancelling")

    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self._result_queue.get_nowait()
                try:
                    self._handle_msg(msg)
                except Exception as exc:
                    import traceback
                    self._log(f"UI error: {exc}", "error")
                    self._log(traceback.format_exc(), "error")
        except queue.Empty:
            pass

        if self._worker_thread is not None:
            self._after_id = self.root.after(80, self._poll_queue)

    def _handle_msg(self, msg: _Msg) -> None:
        if msg.type == _LOG:
            payload = msg.payload
            if payload == "status_cancelled":
                self._log(self._t("status_cancelled"), "warning")
                self._set_status("status_cancelled")
            elif isinstance(payload, dict) and payload.get("key") == "log_result":
                self._log(
                    self._t(payload["key"], **payload.get("kwargs", {})),
                    "normal",
                )
            else:
                self._log(str(payload), "normal")

        elif msg.type == _WARNING:
            self._log(self._t("log_warning", w=msg.payload), "warning")

        elif msg.type == _PROGRESS:
            info: dict = msg.payload
            current = info["current"]
            total = info["total"]
            pct = current / total
            self._progress_bar.set(pct)
            self._progress_label.configure(text=f"{current}/{total}")
            self._progress_text = f"{current}/{total}"
            self._runtime_progress_value.configure(text=self._progress_text)

        elif msg.type == _RESULT:
            info: dict = msg.payload
            source_path = Path(info["filepath"])
            fname = source_path.name
            r = info.get("result")
            error = info.get("error")
            result_output_dir = info.get("output_dir")

            if error:
                self._result_stats["errors"] += 1
                if self._tree is not None:
                    self._tree.insert(
                        "",
                        tk.END,
                        iid=str(source_path),
                        values=(
                            fname, "", "", "", "",
                            f"Error: {error[:50]}",
                        ),
                        tags=("error",),
                    )
            elif r is not None:
                has_warnings = bool(r.warnings)
                if has_warnings:
                    self._result_stats["warnings"] += 1
                else:
                    self._result_stats["ok"] += 1
                status = (
                    self._t("status_ok_warnings")
                    if has_warnings
                    else self._t("status_ok")
                )
                tag = "warning" if has_warnings else "ok"
                means_str = ", ".join(f"{m:.4f}" for m in r.state_means)
                self._classified_outputs[str(source_path)] = self._classified_output_path(
                    source_path,
                    result_output_dir,
                )
                self._last_output_path = str(
                    self._classified_outputs[str(source_path)]
                )
                
                # Store results for the right-side summary panel
                self._results_map[str(source_path)] = r
                self._runtime_summary_value.configure(
                    text=self._t(
                        "runtime_summary_value",
                        ok=self._result_stats["ok"],
                        warnings=self._result_stats["warnings"],
                        errors=self._result_stats["errors"],
                    )
                )
                self._runtime_output_value.configure(text=self._last_output_path)
                if self._tree is not None:
                    self._tree.insert(
                        "",
                        tk.END,
                        iid=str(source_path),
                        values=(
                            fname,
                            r.n_states,
                            f"{r.log_prob:.2f}",
                            means_str,
                            f"{r.state_sigma:.4f}",
                            status,
                        ),
                        tags=(tag,),
                    )
                    self._tree.selection_set(str(source_path))
                    self._tree.focus(str(source_path))
                    self._selected_result_path = str(source_path)
                    self._set_result_summary(str(source_path))
                self._log(
                    self._t("log_output_written", path=self._last_output_path),
                    "success",
                )

        elif msg.type == _DONE:
            self._log(self._t("log_complete"), "success")
            self._log(
                self._t("log_complete_summary", **self._result_stats),
                "header",
            )
            self._runtime_summary_value.configure(
                text=self._t(
                    "runtime_summary_value",
                    ok=self._result_stats["ok"],
                    warnings=self._result_stats["warnings"],
                    errors=self._result_stats["errors"],
                )
            )
            self._set_ui_running(False)
            self._set_status("status_complete")
            self._worker_thread = None
            if self._after_id is not None:
                self.root.after_cancel(self._after_id)
                self._after_id = None
            if self._result_stats["ok"] > 0:
                messagebox.showinfo(
                    self._t("msg_run_complete_title"),
                    self._t(
                        "msg_run_complete",
                        ok=self._result_stats["ok"],
                        warnings=self._result_stats["warnings"],
                        errors=self._result_stats["errors"],
                        path=self._last_output_path or self._t("result_panel_none"),
                    ),
                )

        elif msg.type == _REVIEW_DONE:
            info: dict = msg.payload
            self._review_image_paths = list(info.get("image_paths", []))
            count = int(info.get("count", 0))
            for image_path in self._review_image_paths:
                self._last_output_path = image_path
                self._runtime_output_value.configure(text=image_path)
                self._log(
                    self._t("log_review_output_written", path=image_path),
                    "success",
                )
            if self._review_image_paths:
                messagebox.showinfo(
                    self._t("msg_review_complete_title"),
                    self._t(
                        "msg_review_complete",
                        count=count,
                        paths="\n".join(self._review_image_paths),
                    ),
                )

        elif msg.type == _ERROR:
            self._log(self._t("log_error", e=msg.payload), "error")


def run_gui() -> None:
    import traceback

    def _excepthook(exc_type, exc_value, exc_tb):
        _append_debug_log(
            "Unhandled GUI exception:\n"
            + "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        )
        traceback.print_exception(exc_type, exc_value, exc_tb)
        try:
            messagebox.showerror(
                "FretHMM Error",
                f"{exc_type.__name__}: {exc_value}",
            )
        except Exception:
            pass

    sys.excepthook = _excepthook

    ctk.set_appearance_mode("Light")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    try:
        icon_ico = _resource_path("frethmm/assets/frethmm.ico")
        if icon_ico.exists():
            root.iconbitmap(default=str(icon_ico))
        icon_path = _resource_path("frethmm/assets/frethmm_logo.png")
        if icon_path.exists():
            icon = tk.PhotoImage(file=str(icon_path))
            root.iconphoto(True, icon)
            root._frethmm_icon = icon
    except Exception:
        pass
    root.title("FretHMM — Single-Molecule State Classification")
    root.minsize(1180, 660)

    w, h = 1280, 720
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw - w) // 2
    y = (sh - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")

    fonts = _detect_fonts()

    app = _App(root, fonts)
    app.build()

    def _lazy_init():
        import importlib

        for mod in ("numpy", "scipy", "hmmlearn", "sklearn"):
            try:
                importlib.import_module(mod)
            except Exception:
                pass

    threading.Thread(target=_lazy_init, daemon=True).start()

    root.mainloop()


def main() -> None:
    multiprocessing.freeze_support()
    if "--version" in sys.argv[1:]:
        print(f"FretHMM {_VERSION}")
        return
    run_gui()


if __name__ == "__main__":
    main()
