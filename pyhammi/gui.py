"""Simple GUI for pyHaMMy using tkinter."""

import sys
from pathlib import Path
from typing import Optional

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    HAS_TK = True
except ImportError:
    HAS_TK = False


def run_gui() -> None:
    if not HAS_TK:
        print("tkinter is required for GUI mode.")
        return

    root = tk.Tk()
    root.title("pyHaMMy")
    root.geometry("700x500")
    root.resizable(True, True)

    app = _App(root)
    app.build()
    root.mainloop()


class _App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.selected_files: list[str] = []
        self.output_dir: Optional[str] = None

    def build(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        file_frame = ttk.LabelFrame(main, text="Input Files", padding=5)
        file_frame.pack(fill=tk.X, pady=(0, 5))

        btn_row = ttk.Frame(file_frame)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="Add Files...", command=self._add_files).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_row, text="Add Directory...", command=self._add_dir).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_row, text="Clear", command=self._clear_files).pack(side=tk.LEFT)

        self.file_listbox = tk.Listbox(file_frame, height=5)
        self.file_listbox.pack(fill=tk.X, pady=(5, 0))

        param_frame = ttk.LabelFrame(main, text="HMM Parameters", padding=5)
        param_frame.pack(fill=tk.X, pady=(0, 5))

        row1 = ttk.Frame(param_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Number of states:").pack(side=tk.LEFT)
        self.states_var = tk.IntVar(value=2)
        ttk.Spinbox(row1, from_=1, to=20, textvariable=self.states_var, width=5).pack(side=tk.LEFT, padx=5)

        row2 = ttk.Frame(param_frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Initial guesses (comma-sep):").pack(side=tk.LEFT)
        self.guesses_var = tk.StringVar()
        ttk.Entry(row2, textvariable=self.guesses_var, width=30).pack(side=tk.LEFT, padx=5)

        row3 = ttk.Frame(param_frame)
        row3.pack(fill=tk.X, pady=2)
        ttk.Label(row3, text="Max iterations:").pack(side=tk.LEFT)
        self.iter_var = tk.IntVar(value=500)
        ttk.Spinbox(row3, from_=10, to=10000, textvariable=self.iter_var, width=7).pack(side=tk.LEFT, padx=5)
        ttk.Label(row3, text="Workers:").pack(side=tk.LEFT, padx=(20, 0))
        self.workers_var = tk.IntVar(value=1)
        ttk.Spinbox(row3, from_=1, to=32, textvariable=self.workers_var, width=3).pack(side=tk.LEFT, padx=5)

        row4 = ttk.Frame(param_frame)
        row4.pack(fill=tk.X, pady=2)
        ttk.Label(row4, text="Data mode:").pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="auto")
        ttk.Combobox(row4, textvariable=self.mode_var, width=15,
                     values=["auto", "fret", "donor_acceptor", "single_channel"],
                     state="readonly").pack(side=tk.LEFT, padx=5)

        out_frame = ttk.LabelFrame(main, text="Output", padding=5)
        out_frame.pack(fill=tk.X, pady=(0, 5))
        out_row = ttk.Frame(out_frame)
        out_row.pack(fill=tk.X)
        ttk.Button(out_row, text="Output Dir...", command=self._select_output).pack(side=tk.LEFT, padx=(0, 5))
        self.output_label = ttk.Label(out_row, text="(same as input)")
        self.output_label.pack(side=tk.LEFT)

        run_btn = ttk.Button(main, text="Run Analysis", command=self._run)
        run_btn.pack(pady=10)

        self.log_text = tk.Text(main, height=10, state=tk.DISABLED, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _log(self, msg: str):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def _add_files(self):
        files = filedialog.askopenfilenames(
            title="Select trace files",
            filetypes=[("Data files", "*.dat *.txt *.csv *.tsv"), ("All files", "*.*")],
        )
        for f in files:
            if f not in self.selected_files:
                self.selected_files.append(f)
                self.file_listbox.insert(tk.END, f)

    def _add_dir(self):
        d = filedialog.askdirectory(title="Select directory with trace files")
        if d:
            from pyhammi.io import find_trace_files
            files = find_trace_files(d)
            for f in files:
                s = str(f)
                if s not in self.selected_files:
                    self.selected_files.append(s)
                    self.file_listbox.insert(tk.END, s)

    def _clear_files(self):
        self.selected_files.clear()
        self.file_listbox.delete(0, tk.END)

    def _select_output(self):
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            self.output_dir = d
            self.output_label.config(text=d)

    def _run(self):
        if not self.selected_files:
            messagebox.showwarning("No files", "Please add trace files first.")
            return

        from pyhammi.config import HMMConfig
        from pyhammi.model import process_file

        guesses = None
        g_str = self.guesses_var.get().strip()
        if g_str:
            guesses = [float(g) for g in g_str.split(",")]

        config = HMMConfig(
            n_states=self.states_var.get(),
            max_iter=self.iter_var.get(),
            guesses=guesses,
            workers=self.workers_var.get(),
            data_mode=self.mode_var.get(),
        )
        output_dir = Path(self.output_dir) if self.output_dir else None

        self._log(f"Starting analysis: {len(self.selected_files)} file(s), "
                  f"{config.n_states} states...")

        for i, fp in enumerate(self.selected_files, 1):
            self._log(f"[{i}/{len(self.selected_files)}] {Path(fp).name}")
            try:
                result = process_file(Path(fp), config, output_dir)
                self._log(f"  log_prob={result.log_prob:.2f}, "
                          f"means={result.means}")
            except Exception as e:
                self._log(f"  ERROR: {e}")

        self._log("Done.")
