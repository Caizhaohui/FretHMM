"""tkinter GUI for pyHaMMy — lazy imports for fast startup, i18n, menu bar."""

from __future__ import annotations

import multiprocessing
import os
import platform
import queue
import threading
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Optional

import tkinter as tk

_VERSION = "0.1.0"
try:
    from pyhammi import __version__ as _VERSION
except Exception:
    pass

_LOG = "log"
_PROGRESS = "progress"
_RESULT = "result"
_DONE = "done"
_ERROR = "error"
_WARNING = "warning"


class _Msg:
    __slots__ = ("type", "payload")

    def __init__(self, typ: str, payload: Any = None) -> None:
        self.type = typ
        self.payload = payload


def _worker(
    files: list[Path],
    config_bytes: bytes,
    output_dir: Optional[Path],
    cancel_event: threading.Event,
    result_queue: queue.Queue[_Msg],
) -> None:
    import pickle
    from pyhammi.model import process_file

    config = pickle.loads(config_bytes)
    total = len(files)
    for i, fp in enumerate(files):
        if cancel_event.is_set():
            result_queue.put(_Msg(_LOG, "status_cancelled"))
            break

        result_queue.put(_Msg(_LOG, f"[{i + 1}/{total}] {fp.name}"))
        result_queue.put(_Msg(_PROGRESS, {"current": i + 1, "total": total}))

        try:
            r = process_file(fp, config, output_dir)

            for w in r.warnings:
                result_queue.put(_Msg(_WARNING, w))

            result_queue.put(_Msg(_RESULT, {"filepath": str(fp), "result": r}))
            result_queue.put(
                _Msg(
                    _LOG,
                    "log_result",
                    lp=f"{r.log_prob:.2f}",
                    m=[round(m, 4) for m in r.means.tolist()],
                    sig=f"{r.sigma:.4f}",
                )
            )
        except Exception as exc:
            result_queue.put(_Msg(_ERROR, str(exc)))
            result_queue.put(
                _Msg(
                    _RESULT,
                    {"filepath": str(fp), "result": None, "error": str(exc)},
                )
            )

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


def _configure_styles(root: tk.Tk, fonts: dict) -> ttk.Style:
    style = ttk.Style(root)

    available = style.theme_names()
    if "clam" in available:
        style.theme_use("clam")

    bg = "#f5f5f5"
    fg = "#212121"
    accent = "#1565C0"
    field_bg = "#ffffff"
    select_bg = "#BBDEFB"

    style.configure(".", background=bg, foreground=fg, font=fonts["label"])
    style.configure("TFrame", background=bg)
    style.configure("TLabel", background=bg, foreground=fg, font=fonts["label"])
    style.configure(
        "TLabelframe", background=bg, foreground=fg, font=fonts["heading"]
    )
    style.configure(
        "TLabelframe.Label",
        background=bg,
        foreground=accent,
        font=fonts["heading"],
    )
    style.configure("TButton", font=fonts["button"], padding=(12, 6))
    style.configure("TEntry", font=fonts["entry"], fieldbackground=field_bg)
    style.configure("TSpinbox", font=fonts["entry"], fieldbackground=field_bg)
    style.configure("TCombobox", font=fonts["entry"], fieldbackground=field_bg)

    style.configure(
        "Treeview",
        font=fonts["tree"],
        background=field_bg,
        foreground=fg,
        fieldbackground=field_bg,
        rowheight=28,
    )
    style.configure(
        "Treeview.Heading",
        font=fonts["tree_heading"],
        background="#e0e0e0",
        foreground=fg,
    )
    style.map(
        "Treeview",
        background=[("selected", select_bg)],
        foreground=[("selected", accent)],
    )

    style.configure(
        "Horizontal.TProgressbar",
        troughcolor="#e0e0e0",
        background=accent,
        thickness=22,
    )

    style.configure("Status.TLabel", font=fonts["status"], foreground="#757575")

    style.configure(
        "Accent.TButton",
        font=(fonts["button"][0], 11, "bold"),
        padding=(20, 8),
    )

    return style


class _App:
    def __init__(self, root: tk.Tk, fonts: dict) -> None:
        self.root = root
        self.fonts = fonts
        self.selected_files: list[str] = []
        self.output_dir: Optional[str] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()
        self._result_queue: queue.Queue[_Msg] = queue.Queue()
        self._after_id: Optional[str] = None
        self._lang = "en"

    def _t(self, key: str, **kwargs) -> str:
        from pyhammi.i18n import t, set_language, get_language

        lang = get_language()
        if lang != self._lang:
            self._lang = lang
        return t(key, **kwargs)

    def build(self) -> None:
        self._build_menu()

        main = ttk.Frame(self.root, padding=(16, 12, 16, 8))
        main.pack(fill=tk.BOTH, expand=True)

        title_frame = ttk.Frame(main)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        self._title_label = ttk.Label(
            title_frame,
            text="pyHaMMy",
            font=self.fonts["title"],
            foreground="#1565C0",
        )
        self._title_label.pack(side=tk.LEFT)
        self._subtitle_label = ttk.Label(
            title_frame,
            text=f"  {self._t('subtitle')}  v{_VERSION}",
            font=self.fonts["label"],
            foreground="#757575",
        )
        self._subtitle_label.pack(side=tk.LEFT, padx=(8, 0))

        self._build_files_section(main)
        self._build_params_section(main)
        self._build_output_section(main)
        self._build_action_section(main)
        self._build_results_section(main)
        self._build_log_section(main)
        self._build_status_bar(main)

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
        menubar.add_cascade(label=self._t("menu_settings"), menu=settings_menu)
        self._settings_menu = settings_menu
        self._lang_menu = lang_menu

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(
            label=self._t("menu_help_about"), command=self._show_about
        )
        menubar.add_cascade(label=self._t("menu_help"), menu=help_menu)
        self._help_menu = help_menu

        self.root.config(menu=menubar)
        self._menubar = menubar

    def _build_files_section(self, parent: ttk.Frame) -> None:
        self._file_frame = ttk.LabelFrame(
            parent, text=f" {self._t('section_files')} ", padding=(10, 8)
        )
        self._file_frame.pack(fill=tk.X, pady=(0, 8))

        btn_row = ttk.Frame(self._file_frame)
        btn_row.pack(fill=tk.X)
        self._btn_add = ttk.Button(
            btn_row, text=self._t("btn_add_files"), command=self._add_files
        )
        self._btn_add.pack(side=tk.LEFT, padx=(0, 6))
        self._btn_folder = ttk.Button(
            btn_row, text=self._t("btn_add_folder"), command=self._add_folder
        )
        self._btn_folder.pack(side=tk.LEFT, padx=(0, 6))
        self._btn_clear = ttk.Button(
            btn_row, text=self._t("btn_clear"), command=self._clear_files
        )
        self._btn_clear.pack(side=tk.LEFT)

        list_frame = ttk.Frame(self._file_frame)
        list_frame.pack(fill=tk.X, pady=(6, 0))
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self._file_listbox = tk.Listbox(
            list_frame,
            height=4,
            selectmode=tk.EXTENDED,
            yscrollcommand=scrollbar.set,
            font=self.fonts["entry"],
            bg="#ffffff",
            selectbackground="#BBDEFB",
            selectforeground="#1565C0",
            relief=tk.FLAT,
            highlightthickness=1,
            highlightcolor="#90CAF9",
            highlightbackground="#bdbdbd",
        )
        scrollbar.config(command=self._file_listbox.yview)
        self._file_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._file_listbox.bind("<Delete>", self._delete_selected_files)

    def _build_params_section(self, parent: ttk.Frame) -> None:
        self._param_frame = ttk.LabelFrame(
            parent, text=f" {self._t('section_params')} ", padding=(10, 8)
        )
        self._param_frame.pack(fill=tk.X, pady=(0, 8))

        row1 = ttk.Frame(self._param_frame)
        row1.pack(fill=tk.X, pady=3)
        self._lbl_states = ttk.Label(row1, text=self._t("label_states"))
        self._lbl_states.pack(side=tk.LEFT)
        self._states_var = tk.IntVar(value=2)
        ttk.Spinbox(
            row1, from_=1, to=30, textvariable=self._states_var, width=5
        ).pack(side=tk.LEFT, padx=(4, 16))
        self._lbl_guesses = ttk.Label(
            row1, text=self._t("label_guesses")
        )
        self._lbl_guesses.pack(side=tk.LEFT)
        self._guesses_var = tk.StringVar()
        ttk.Entry(row1, textvariable=self._guesses_var, width=28).pack(
            side=tk.LEFT, padx=(4, 0)
        )

        row2 = ttk.Frame(self._param_frame)
        row2.pack(fill=tk.X, pady=3)
        self._lbl_iter = ttk.Label(row2, text=self._t("label_max_iter"))
        self._lbl_iter.pack(side=tk.LEFT)
        self._iter_var = tk.IntVar(value=500)
        ttk.Spinbox(
            row2, from_=10, to=100000, textvariable=self._iter_var, width=7
        ).pack(side=tk.LEFT, padx=(4, 16))
        self._lbl_tol = ttk.Label(row2, text=self._t("label_tolerance"))
        self._lbl_tol.pack(side=tk.LEFT)
        self._tol_var = tk.StringVar(value="1e-4")
        ttk.Entry(row2, textvariable=self._tol_var, width=10).pack(
            side=tk.LEFT, padx=(4, 16)
        )
        self._lbl_workers = ttk.Label(row2, text=self._t("label_workers"))
        self._lbl_workers.pack(side=tk.LEFT)
        self._workers_var = tk.IntVar(value=1)
        ttk.Spinbox(
            row2, from_=1, to=32, textvariable=self._workers_var, width=4
        ).pack(side=tk.LEFT, padx=(4, 0))

        row3 = ttk.Frame(self._param_frame)
        row3.pack(fill=tk.X, pady=3)
        self._lbl_mode = ttk.Label(row3, text=self._t("label_data_mode"))
        self._lbl_mode.pack(side=tk.LEFT)
        self._mode_var = tk.StringVar(value="auto")
        ttk.Combobox(
            row3,
            textvariable=self._mode_var,
            width=18,
            values=["auto", "fret", "donor_acceptor", "single_channel"],
            state="readonly",
        ).pack(side=tk.LEFT, padx=(4, 0))

    def _build_output_section(self, parent: ttk.Frame) -> None:
        self._out_frame = ttk.LabelFrame(
            parent, text=f" {self._t('section_output')} ", padding=(10, 8)
        )
        self._out_frame.pack(fill=tk.X, pady=(0, 8))
        out_row = ttk.Frame(self._out_frame)
        out_row.pack(fill=tk.X)
        self._btn_output = ttk.Button(
            out_row, text=self._t("btn_output_folder"), command=self._select_output
        )
        self._btn_output.pack(side=tk.LEFT, padx=(0, 8))
        self._output_label = ttk.Label(
            out_row,
            text=self._t("output_same_as_input"),
            foreground="#757575",
        )
        self._output_label.pack(side=tk.LEFT)

    def _build_action_section(self, parent: ttk.Frame) -> None:
        self._act_frame = ttk.Frame(parent)
        self._act_frame.pack(fill=tk.X, pady=(0, 8))

        self._run_btn = ttk.Button(
            self._act_frame,
            text=self._t("btn_run"),
            command=self._run,
            style="Accent.TButton",
        )
        self._run_btn.pack(side=tk.LEFT, padx=(0, 8))
        self._cancel_btn = ttk.Button(
            self._act_frame,
            text=self._t("btn_cancel"),
            command=self._cancel,
            state=tk.DISABLED,
        )
        self._cancel_btn.pack(side=tk.LEFT, padx=(0, 16))

        self._progress_bar = ttk.Progressbar(
            self._act_frame,
            maximum=100,
            mode="determinate",
            style="Horizontal.TProgressbar",
        )
        self._progress_bar.pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8)
        )
        self._progress_label = ttk.Label(
            self._act_frame, text="", width=10, anchor=tk.E
        )
        self._progress_label.pack(side=tk.RIGHT)

    def _build_results_section(self, parent: ttk.Frame) -> None:
        self._results_frame = ttk.LabelFrame(
            parent, text=f" {self._t('section_results')} ", padding=(10, 8)
        )
        self._results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        columns = ("file", "states", "log_prob", "means", "sigma", "status")
        self._tree = ttk.Treeview(
            self._results_frame,
            columns=columns,
            show="headings",
            height=6,
        )
        self._tree.heading("file", text=self._t("col_file"))
        self._tree.heading("states", text=self._t("col_states"))
        self._tree.heading("log_prob", text=self._t("col_log_prob"))
        self._tree.heading("means", text=self._t("col_means"))
        self._tree.heading("sigma", text=self._t("col_sigma"))
        self._tree.heading("status", text=self._t("col_status"))

        self._tree.column("file", width=180, minwidth=100)
        self._tree.column("states", width=60, minwidth=50, anchor=tk.CENTER)
        self._tree.column("log_prob", width=100, minwidth=70, anchor=tk.CENTER)
        self._tree.column("means", width=180, minwidth=100)
        self._tree.column("sigma", width=80, minwidth=60, anchor=tk.CENTER)
        self._tree.column("status", width=100, minwidth=60, anchor=tk.CENTER)

        tree_scroll = ttk.Scrollbar(
            self._results_frame, orient=tk.VERTICAL, command=self._tree.yview
        )
        self._tree.configure(yscrollcommand=tree_scroll.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._tree.tag_configure("ok", foreground="#2E7D32")
        self._tree.tag_configure("warning", foreground="#E65100")
        self._tree.tag_configure("error", foreground="#C62828")

    def _build_log_section(self, parent: ttk.Frame) -> None:
        self._log_frame = ttk.LabelFrame(
            parent, text=f" {self._t('section_log')} ", padding=(10, 8)
        )
        self._log_frame.pack(fill=tk.BOTH, expand=True)

        self._log_text = tk.Text(
            self._log_frame,
            height=6,
            state=tk.DISABLED,
            font=self.fonts["log"],
            wrap=tk.WORD,
            bg="#ffffff",
            relief=tk.FLAT,
            highlightthickness=1,
            highlightcolor="#90CAF9",
            highlightbackground="#bdbdbd",
            padx=8,
            pady=6,
        )
        self._log_text.tag_configure("normal", foreground="#212121")
        self._log_text.tag_configure("warning", foreground="#E65100")
        self._log_text.tag_configure("error", foreground="#C62828")
        self._log_text.tag_configure("success", foreground="#2E7D32")
        self._log_text.tag_configure("header", foreground="#1565C0")

        log_scroll = ttk.Scrollbar(
            self._log_frame, orient=tk.VERTICAL, command=self._log_text.yview
        )
        self._log_text.configure(yscrollcommand=log_scroll.set)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_status_bar(self, parent: ttk.Frame) -> None:
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(6, 0))
        self._status_label = ttk.Label(
            status_frame, text=self._t("status_ready"), style="Status.TLabel"
        )
        self._status_label.pack(side=tk.LEFT)
        ttk.Label(
            status_frame, text=f"v{_VERSION}", style="Status.TLabel"
        ).pack(side=tk.RIGHT)

    def _switch_language(self, lang: str) -> None:
        from pyhammi.i18n import set_language

        set_language(lang)
        self._lang = lang
        self._refresh_all_text()

    def _refresh_all_text(self) -> None:
        self.root.title(self._t("app_title"))
        self._subtitle_label.config(
            text=f"  {self._t('subtitle')}  v{_VERSION}"
        )

        self._file_frame.config(text=f" {self._t('section_files')} ")
        self._btn_add.config(text=self._t("btn_add_files"))
        self._btn_folder.config(text=self._t("btn_add_folder"))
        self._btn_clear.config(text=self._t("btn_clear"))

        self._param_frame.config(text=f" {self._t('section_params')} ")
        self._lbl_states.config(text=self._t("label_states"))
        self._lbl_guesses.config(text=self._t("label_guesses"))
        self._lbl_iter.config(text=self._t("label_max_iter"))
        self._lbl_tol.config(text=self._t("label_tolerance"))
        self._lbl_workers.config(text=self._t("label_workers"))
        self._lbl_mode.config(text=self._t("label_data_mode"))

        self._out_frame.config(text=f" {self._t('section_output')} ")
        self._btn_output.config(text=self._t("btn_output_folder"))
        if not self.output_dir:
            self._output_label.config(text=self._t("output_same_as_input"))

        self._run_btn.config(text=self._t("btn_run"))
        self._cancel_btn.config(text=self._t("btn_cancel"))

        self._results_frame.config(text=f" {self._t('section_results')} ")
        self._tree.heading("file", text=self._t("col_file"))
        self._tree.heading("states", text=self._t("col_states"))
        self._tree.heading("log_prob", text=self._t("col_log_prob"))
        self._tree.heading("means", text=self._t("col_means"))
        self._tree.heading("sigma", text=self._t("col_sigma"))
        self._tree.heading("status", text=self._t("col_status"))

        self._log_frame.config(text=f" {self._t('section_log')} ")

        self._status_label.config(text=self._t("status_ready"))

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
        self._lang_menu.entryconfig(0, label=self._t("menu_settings_lang_en"))
        self._lang_menu.entryconfig(1, label=self._t("menu_settings_lang_zh"))

        self._help_menu.entryconfig(0, label=self._t("menu_help_about"))

        self._menubar.entryconfig(0, label=self._t("menu_file"))
        self._menubar.entryconfig(1, label=self._t("menu_settings"))
        self._menubar.entryconfig(2, label=self._t("menu_help"))

    def _show_params_dialog(self) -> None:
        dlg = tk.Toplevel(self.root)
        dlg.title(self._t("dlg_params_title"))
        dlg.geometry("420x320")
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set()

        sw = dlg.winfo_screenwidth()
        sh = dlg.winfo_screenheight()
        x = (sw - 420) // 2
        y = (sh - 320) // 2
        dlg.geometry(f"420x320+{x}+{y}")

        frame = ttk.Frame(dlg, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text=self._t("label_states")).grid(
            row=0, column=0, sticky=tk.W, pady=6
        )
        states_var = tk.IntVar(value=self._states_var.get())
        ttk.Spinbox(
            frame, from_=1, to=30, textvariable=states_var, width=8
        ).grid(row=0, column=1, padx=(10, 0), pady=6)

        ttk.Label(frame, text=self._t("label_max_iter")).grid(
            row=1, column=0, sticky=tk.W, pady=6
        )
        iter_var = tk.IntVar(value=self._iter_var.get())
        ttk.Spinbox(
            frame, from_=10, to=100000, textvariable=iter_var, width=8
        ).grid(row=1, column=1, padx=(10, 0), pady=6)

        ttk.Label(frame, text=self._t("label_tolerance")).grid(
            row=2, column=0, sticky=tk.W, pady=6
        )
        tol_var = tk.StringVar(value=self._tol_var.get())
        ttk.Entry(frame, textvariable=tol_var, width=10).grid(
            row=2, column=1, padx=(10, 0), pady=6
        )

        ttk.Label(frame, text=self._t("label_workers")).grid(
            row=3, column=0, sticky=tk.W, pady=6
        )
        workers_var = tk.IntVar(value=self._workers_var.get())
        ttk.Spinbox(
            frame, from_=1, to=32, textvariable=workers_var, width=8
        ).grid(row=3, column=1, padx=(10, 0), pady=6)

        ttk.Label(frame, text=self._t("label_data_mode")).grid(
            row=4, column=0, sticky=tk.W, pady=6
        )
        mode_var = tk.StringVar(value=self._mode_var.get())
        ttk.Combobox(
            frame,
            textvariable=mode_var,
            width=16,
            values=["auto", "fret", "donor_acceptor", "single_channel"],
            state="readonly",
        ).grid(row=4, column=1, padx=(10, 0), pady=6)

        ttk.Label(frame, text=self._t("label_guesses")).grid(
            row=5, column=0, sticky=tk.W, pady=6
        )
        guesses_var = tk.StringVar(value=self._guesses_var.get())
        ttk.Entry(frame, textvariable=guesses_var, width=28).grid(
            row=5, column=1, padx=(10, 0), pady=6
        )

        def apply():
            self._states_var.set(states_var.get())
            self._iter_var.set(iter_var.get())
            self._tol_var.set(tol_var.get())
            self._workers_var.set(workers_var.get())
            self._mode_var.set(mode_var.get())
            self._guesses_var.set(guesses_var.get())
            dlg.destroy()

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=6, column=0, columnspan=2, pady=(16, 0))
        ttk.Button(btn_frame, text="OK", command=apply, width=12).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(
            btn_frame, text="Cancel", command=dlg.destroy, width=12
        ).pack(side=tk.LEFT, padx=6)

    def _show_about(self) -> None:
        messagebox.showinfo(
            self._t("about_title"),
            self._t("about_msg", v=_VERSION),
        )

    def _log(self, msg: str, tag: str = "normal") -> None:
        self._log_text.config(state=tk.NORMAL)
        self._log_text.insert(tk.END, msg + "\n", tag)
        self._log_text.see(tk.END)
        self._log_text.config(state=tk.DISABLED)

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
            self._status_label.config(
                text=self._t("status_files_selected", n=len(self.selected_files))
            )

    def _add_folder(self) -> None:
        d = filedialog.askdirectory(
            title=self._t("folder_dialog_title")
        )
        if not d:
            return
        from pyhammi.io import find_trace_files

        found = find_trace_files(d)
        for p in found:
            s = str(p)
            if s not in self.selected_files:
                self.selected_files.append(s)
                self._file_listbox.insert(tk.END, s)
        if not found:
            self._log(self._t("msg_no_traces"), "warning")
        else:
            self._status_label.config(
                text=self._t(
                    "status_files_selected", n=len(self.selected_files)
                )
            )

    def _clear_files(self) -> None:
        self.selected_files.clear()
        self._file_listbox.delete(0, tk.END)
        self._status_label.config(text=self._t("status_ready"))

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
        self._status_label.config(
            text=self._t("status_files_selected", n=len(self.selected_files))
        )

    def _select_output(self) -> None:
        d = filedialog.askdirectory(
            title=self._t("output_dialog_title")
        )
        if d:
            self.output_dir = d
            self._output_label.config(text=d, foreground="#212121")

    def _build_config(self):
        from pyhammi.config import HMMConfig

        guesses = None
        g_str = self._guesses_var.get().strip()
        if g_str:
            guesses = [float(g) for g in g_str.split(",")]

        tol_str = self._tol_var.get().strip()
        try:
            tol = float(tol_str)
        except ValueError:
            raise ValueError(self._t("msg_invalid_tol", v=tol_str))

        n_states = self._states_var.get()

        if guesses is not None and len(guesses) != n_states:
            raise ValueError(
                self._t(
                    "msg_guess_mismatch", g=len(guesses), s=n_states
                )
            )

        return HMMConfig(
            n_states=n_states,
            max_iter=self._iter_var.get(),
            tol=tol,
            guesses=guesses if guesses else None,
            workers=self._workers_var.get(),
            data_mode=self._mode_var.get(),
        )

    def _set_ui_running(self, running: bool) -> None:
        if running:
            self._run_btn.config(state=tk.DISABLED)
            self._cancel_btn.config(state=tk.NORMAL)
            self._status_label.config(text=self._t("status_running"))
        else:
            self._run_btn.config(state=tk.NORMAL)
            self._cancel_btn.config(state=tk.DISABLED)
            self._progress_bar["value"] = 0
            self._progress_label.config(text="")

    def _run(self) -> None:
        if not self.selected_files:
            messagebox.showwarning(
                self._t("msg_no_files_title"), self._t("msg_no_files")
            )
            return

        try:
            config = self._build_config()
        except ValueError as e:
            messagebox.showerror(self._t("msg_invalid_params"), str(e))
            return

        import pickle

        output_dir = Path(self.output_dir) if self.output_dir else None
        files = [Path(p) for p in self.selected_files]
        config_bytes = pickle.dumps(config)

        for item in self._tree.get_children():
            self._tree.delete(item)

        self._set_ui_running(True)
        self._cancel_event.clear()
        self._result_queue = queue.Queue()

        self._log(
            self._t(
                "log_starting", n=len(files), s=config.n_states
            ),
            "header",
        )

        self._worker_thread = threading.Thread(
            target=_worker,
            args=(
                files,
                config_bytes,
                output_dir,
                self._cancel_event,
                self._result_queue,
            ),
            daemon=True,
        )
        self._worker_thread.start()
        self._poll_queue()

    def _cancel(self) -> None:
        self._cancel_event.set()
        self._log(self._t("log_cancelling"), "warning")
        self._status_label.config(text=self._t("status_cancelling"))

    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self._result_queue.get_nowait()
                self._handle_msg(msg)
        except queue.Empty:
            pass

        self._after_id = self.root.after(80, self._poll_queue)

    def _handle_msg(self, msg: _Msg) -> None:
        if msg.type == _LOG:
            payload = msg.payload
            if payload == "status_cancelled":
                self._log(self._t("status_cancelled"), "warning")
            elif isinstance(payload, str) and payload == "log_result":
                pass
            else:
                self._log(str(payload), "normal")

        elif msg.type == _WARNING:
            self._log(self._t("log_warning", w=msg.payload), "warning")

        elif msg.type == _PROGRESS:
            info: dict = msg.payload
            current = info["current"]
            total = info["total"]
            pct = int(current / total * 100)
            self._progress_bar["value"] = pct
            self._progress_label.config(text=f"{current}/{total}")

        elif msg.type == _RESULT:
            info: dict = msg.payload
            fname = Path(info["filepath"]).name
            r = info.get("result")
            error = info.get("error")

            if error:
                self._tree.insert(
                    "",
                    tk.END,
                    values=(
                        fname, "", "", "", "",
                        f"Error: {error[:50]}",
                    ),
                    tags=("error",),
                )
            elif r is not None:
                has_warnings = bool(r.warnings)
                status = (
                    self._t("status_ok_warnings")
                    if has_warnings
                    else self._t("status_ok")
                )
                tag = "warning" if has_warnings else "ok"
                means_str = ", ".join(f"{m:.4f}" for m in r.means)
                self._tree.insert(
                    "",
                    tk.END,
                    values=(
                        fname,
                        r.n_states,
                        f"{r.log_prob:.2f}",
                        means_str,
                        f"{r.sigma:.4f}",
                        status,
                    ),
                    tags=(tag,),
                )

        elif msg.type == _DONE:
            self._log(self._t("log_complete"), "success")
            self._set_ui_running(False)
            self._status_label.config(text=self._t("status_complete"))
            self._worker_thread = None
            if self._after_id is not None:
                self.root.after_cancel(self._after_id)
                self._after_id = None

        elif msg.type == _ERROR:
            self._log(self._t("log_error", e=msg.payload), "error")


def run_gui() -> None:
    root = tk.Tk()
    root.title("pyHaMMy — Single-Molecule HMM Analysis")
    root.minsize(750, 550)

    w, h = 900, 720
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw - w) // 2
    y = (sh - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")

    fonts = _detect_fonts()
    _configure_styles(root, fonts)

    app = _App(root, fonts)
    app.build()

    def _lazy_init():
        import importlib

        try:
            importlib.import_module("numpy")
        except Exception:
            pass

    threading.Thread(target=_lazy_init, daemon=True).start()

    root.mainloop()


def main() -> None:
    multiprocessing.freeze_support()
    run_gui()


if __name__ == "__main__":
    main()
