"""Complete tkinter GUI for pyHaMMy — threaded HMM analysis with progress, cancel, and results table."""

from __future__ import annotations

import multiprocessing
import queue
import threading
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Optional

import tkinter as tk

from pyhammi.config import HMMConfig
from pyhammi.model import process_file


# ---------------------------------------------------------------------------
# Worker → UI message types
# ---------------------------------------------------------------------------

_LOG = "log"
_PROGRESS = "progress"
_RESULT = "result"
_DONE = "done"
_ERROR = "error"


class _Msg:
    __slots__ = ("type", "payload")

    def __init__(self, typ: str, payload: Any = None) -> None:
        self.type = typ
        self.payload = payload


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

def _worker(
    files: list[Path],
    config: HMMConfig,
    output_dir: Optional[Path],
    cancel_event: threading.Event,
    result_queue: queue.Queue[_Msg],
) -> None:
    total = len(files)
    for i, fp in enumerate(files):
        if cancel_event.is_set():
            result_queue.put(_Msg(_LOG, "Cancelled by user."))
            break

        result_queue.put(_Msg(_LOG, f"[{i + 1}/{total}] {fp.name}"))
        result_queue.put(_Msg(_PROGRESS, {"current": i + 1, "total": total}))

        try:
            r = process_file(fp, config, output_dir)
            result_queue.put(_Msg(_RESULT, {"filepath": str(fp), "result": r}))
            result_queue.put(
                _Msg(
                    _LOG,
                    f"  log_prob={r.log_prob:.2f}, means={[round(m, 4) for m in r.means.tolist()]}, sigma={r.sigma:.4f}",
                )
            )
        except Exception as exc:
            result_queue.put(
                _Msg(_LOG, f"  ERROR: {exc}\n{traceback.format_exc()}")
            )
            result_queue.put(
                _Msg(
                    _RESULT,
                    {
                        "filepath": str(fp),
                        "result": None,
                        "error": str(exc),
                    },
                )
            )

    result_queue.put(_Msg(_DONE, None))


# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------

class _App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.selected_files: list[str] = []
        self.output_dir: Optional[str] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()
        self._result_queue: queue.Queue[_Msg] = queue.Queue()

        self._after_id: Optional[str] = None

    # -- build UI -----------------------------------------------------------

    def build(self) -> None:
        main = ttk.Frame(self.root, padding=(10, 10, 10, 10))
        main.pack(fill=tk.BOTH, expand=True)

        # ----- file selection -----
        file_frame = ttk.LabelFrame(main, text="Input Files", padding=5)
        file_frame.pack(fill=tk.X, pady=(0, 5))

        btn_row = ttk.Frame(file_frame)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="Add Files...", command=self._add_files).pack(
            side=tk.LEFT, padx=(0, 5)
        )
        ttk.Button(btn_row, text="Add Folder...", command=self._add_folder).pack(
            side=tk.LEFT, padx=(0, 5)
        )
        ttk.Button(btn_row, text="Clear All", command=self._clear_files).pack(
            side=tk.LEFT
        )

        list_frame = ttk.Frame(file_frame)
        list_frame.pack(fill=tk.X, pady=(5, 0))
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self._file_listbox = tk.Listbox(
            list_frame,
            height=5,
            selectmode=tk.EXTENDED,
            yscrollcommand=scrollbar.set,
        )
        scrollbar.config(command=self._file_listbox.yview)
        self._file_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._file_listbox.bind("<Delete>", self._delete_selected_files)

        # ----- parameters -----
        param_frame = ttk.LabelFrame(main, text="HMM Parameters", padding=5)
        param_frame.pack(fill=tk.X, pady=(0, 5))

        row1 = ttk.Frame(param_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="States:").pack(side=tk.LEFT)
        self._states_var = tk.IntVar(value=2)
        ttk.Spinbox(
            row1, from_=1, to=30, textvariable=self._states_var, width=5
        ).pack(side=tk.LEFT, padx=5)

        ttk.Label(row1, text="Initial guesses (comma-separated):").pack(
            side=tk.LEFT, padx=(15, 0)
        )
        self._guesses_var = tk.StringVar()
        ttk.Entry(row1, textvariable=self._guesses_var, width=30).pack(
            side=tk.LEFT, padx=5
        )

        row2 = ttk.Frame(param_frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Max iterations:").pack(side=tk.LEFT)
        self._iter_var = tk.IntVar(value=500)
        ttk.Spinbox(
            row2, from_=10, to=100000, textvariable=self._iter_var, width=7
        ).pack(side=tk.LEFT, padx=5)

        ttk.Label(row2, text="Tolerance:").pack(side=tk.LEFT, padx=(15, 0))
        self._tol_var = tk.StringVar(value="1e-4")
        ttk.Entry(row2, textvariable=self._tol_var, width=10).pack(
            side=tk.LEFT, padx=5
        )

        ttk.Label(row2, text="Workers:").pack(side=tk.LEFT, padx=(15, 0))
        self._workers_var = tk.IntVar(value=1)
        ttk.Spinbox(
            row2, from_=1, to=32, textvariable=self._workers_var, width=4
        ).pack(side=tk.LEFT, padx=5)

        row3 = ttk.Frame(param_frame)
        row3.pack(fill=tk.X, pady=2)
        ttk.Label(row3, text="Data mode:").pack(side=tk.LEFT)
        self._mode_var = tk.StringVar(value="auto")
        ttk.Combobox(
            row3,
            textvariable=self._mode_var,
            width=16,
            values=["auto", "fret", "donor_acceptor", "single_channel"],
            state="readonly",
        ).pack(side=tk.LEFT, padx=5)

        # ----- output -----
        out_frame = ttk.LabelFrame(main, text="Output", padding=5)
        out_frame.pack(fill=tk.X, pady=(0, 5))
        out_row = ttk.Frame(out_frame)
        out_row.pack(fill=tk.X)
        ttk.Button(out_row, text="Output Folder...", command=self._select_output).pack(
            side=tk.LEFT, padx=(0, 5)
        )
        self._output_label = ttk.Label(out_row, text="(same as input file)")
        self._output_label.pack(side=tk.LEFT)

        # ----- progress bar -----
        prog_frame = ttk.Frame(main)
        prog_frame.pack(fill=tk.X, pady=(0, 5))
        self._progress_var = tk.IntVar(value=0)
        self._progress_bar = ttk.Progressbar(
            prog_frame,
            variable=self._progress_var,
            maximum=100,
            mode="determinate",
        )
        self._progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self._progress_label = ttk.Label(prog_frame, text="")
        self._progress_label.pack(side=tk.RIGHT)

        # ----- action buttons -----
        act_row = ttk.Frame(main)
        act_row.pack(fill=tk.X, pady=(0, 5))
        self._run_btn = ttk.Button(
            act_row, text="Run Analysis", command=self._run
        )
        self._run_btn.pack(side=tk.LEFT, padx=(0, 5))
        self._cancel_btn = ttk.Button(
            act_row, text="Cancel", command=self._cancel, state=tk.DISABLED
        )
        self._cancel_btn.pack(side=tk.LEFT)

        # ----- results table -----
        results_frame = ttk.LabelFrame(main, text="Results", padding=5)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        columns = ("file", "states", "log_prob", "means", "sigma", "status")
        self._tree = ttk.Treeview(
            results_frame,
            columns=columns,
            show="headings",
            height=6,
        )
        self._tree.heading("file", text="File")
        self._tree.heading("states", text="#States")
        self._tree.heading("log_prob", text="Log Prob")
        self._tree.heading("means", text="Peak Means")
        self._tree.heading("sigma", text="Sigma")
        self._tree.heading("status", text="Status")

        self._tree.column("file", width=200, minwidth=100)
        self._tree.column("states", width=60, minwidth=50, anchor=tk.CENTER)
        self._tree.column("log_prob", width=100, minwidth=70, anchor=tk.CENTER)
        self._tree.column("means", width=160, minwidth=100)
        self._tree.column("sigma", width=80, minwidth=60, anchor=tk.CENTER)
        self._tree.column("status", width=80, minwidth=60, anchor=tk.CENTER)

        tree_scroll = ttk.Scrollbar(
            results_frame, orient=tk.VERTICAL, command=self._tree.yview
        )
        self._tree.configure(yscrollcommand=tree_scroll.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # ----- log -----
        log_frame = ttk.LabelFrame(main, text="Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self._log_text = tk.Text(
            log_frame,
            height=8,
            state=tk.DISABLED,
            font=("Consolas", 9),
            wrap=tk.WORD,
        )
        log_scroll = ttk.Scrollbar(
            log_frame, orient=tk.VERTICAL, command=self._log_text.yview
        )
        self._log_text.configure(yscrollcommand=log_scroll.set)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # -- logging ------------------------------------------------------------

    def _log(self, msg: str) -> None:
        self._log_text.config(state=tk.NORMAL)
        self._log_text.insert(tk.END, msg + "\n")
        self._log_text.see(tk.END)
        self._log_text.config(state=tk.DISABLED)

    # -- file operations ----------------------------------------------------

    def _add_files(self) -> None:
        files = filedialog.askopenfilenames(
            title="Select trace files",
            filetypes=[
                ("Data files", "*.dat *.txt *.csv *.tsv"),
                ("All files", "*.*"),
            ],
        )
        for f in files:
            if f not in self.selected_files:
                self.selected_files.append(f)
                self._file_listbox.insert(tk.END, f)

    def _add_folder(self) -> None:
        d = filedialog.askdirectory(title="Select directory with trace files")
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
            self._log("No trace files found in selected folder.")

    def _clear_files(self) -> None:
        self.selected_files.clear()
        self._file_listbox.delete(0, tk.END)

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

    # -- output selection ---------------------------------------------------

    def _select_output(self) -> None:
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            self.output_dir = d
            self._output_label.config(text=d)

    # -- validate & run -----------------------------------------------------

    def _build_config(self) -> HMMConfig:
        guesses = None
        g_str = self._guesses_var.get().strip()
        if g_str:
            guesses = [float(g) for g in g_str.split(",")]

        tol_str = self._tol_var.get().strip()
        try:
            tol = float(tol_str)
        except ValueError:
            raise ValueError(f"Invalid tolerance value: {tol_str!r}")

        n_states = self._states_var.get()

        if guesses is not None and len(guesses) != n_states:
            raise ValueError(
                f"Number of guesses ({len(guesses)}) does not match "
                f"number of states ({n_states})."
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
        else:
            self._run_btn.config(state=tk.NORMAL)
            self._cancel_btn.config(state=tk.DISABLED)
            self._progress_var.set(0)
            self._progress_label.config(text="")

    def _run(self) -> None:
        if not self.selected_files:
            messagebox.showwarning("No files", "Please add trace files first.")
            return

        try:
            config = self._build_config()
        except ValueError as e:
            messagebox.showerror("Invalid parameters", str(e))
            return

        output_dir = Path(self.output_dir) if self.output_dir else None
        files = [Path(p) for p in self.selected_files]

        # clear previous results
        for item in self._tree.get_children():
            self._tree.delete(item)

        self._set_ui_running(True)
        self._cancel_event.clear()
        self._result_queue = queue.Queue()

        self._log(f"Starting analysis: {len(files)} file(s), {config.n_states} states...")

        self._worker_thread = threading.Thread(
            target=_worker,
            args=(files, config, output_dir, self._cancel_event, self._result_queue),
            daemon=True,
        )
        self._worker_thread.start()
        self._poll_queue()

    def _cancel(self) -> None:
        self._cancel_event.set()
        self._log("Cancelling...")

    # -- queue polling ------------------------------------------------------

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
            self._log(str(msg.payload))

        elif msg.type == _PROGRESS:
            info: dict = msg.payload
            current = info["current"]
            total = info["total"]
            pct = int(current / total * 100)
            self._progress_var.set(pct)
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
                    values=(fname, "", "", "", "", error),
                )
            elif r is not None:
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
                        "OK",
                    ),
                )

        elif msg.type == _DONE:
            self._log("Analysis complete.")
            self._set_ui_running(False)
            self._worker_thread = None
            if self._after_id is not None:
                self.root.after_cancel(self._after_id)
                self._after_id = None

        elif msg.type == _ERROR:
            self._log(f"FATAL: {msg.payload}")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_gui() -> None:
    """Launch the pyHaMMy GUI.  Safe to call from any module or CLI."""
    root = tk.Tk()
    root.title("pyHaMMy — Single-Molecule HMM Analysis")
    root.minsize(700, 500)

    # default size
    w, h = 820, 620
    # centre on screen
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw - w) // 2
    y = (sh - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")

    # modern theme
    style = ttk.Style(root)
    available = style.theme_names()
    if "clam" in available:
        style.theme_use("clam")

    app = _App(root)
    app.build()
    root.mainloop()


def main() -> None:
    """Entry point for PyInstaller / frozen executables."""
    multiprocessing.freeze_support()
    run_gui()


if __name__ == "__main__":
    main()
