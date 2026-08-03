#!/usr/bin/env python
# coding: utf-8

# In[8]:


"""
CIU Data Extractor and Processor v1.6
By Ismayil Safarli

Requirements: pip install pyopenms pandas numpy
Also requires msconvert.exe (ProteoWizard), to be downloaded separately from https://proteowizard.sourceforge.io/download.html
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
from pathlib import Path
import threading
import glob
import os
import re
import json


import numpy as np
import pandas as pd
from pyopenms import MSExperiment, MzMLFile

try:
    _base_dir = Path(__file__).parent
except NameError:
    _base_dir = Path.cwd()

CONFIG_FILE = _base_dir / "msconvert_config.txt"
SESSION_FILE = _base_dir / "ciu_session.json"


# ============================================================
# Backend logic
# ============================================================

def find_msconvert():
    """Try common install locations for msconvert.exe"""
    candidates = []
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        candidates.extend(glob.glob(
            os.path.join(local_app_data, "Apps", "ProteoWizard*", "msconvert.exe")
        ))
    program_files = [os.environ.get("PROGRAMFILES", ""), os.environ.get("PROGRAMFILES(X86)", "")]
    for pf in program_files:
        if pf:
            candidates.extend(glob.glob(os.path.join(pf, "ProteoWizard*", "msconvert.exe")))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def load_saved_msconvert_path():
    if CONFIG_FILE.exists():
        try:
            path = CONFIG_FILE.read_text(encoding="utf-8").strip()
            if path and os.path.isfile(path):
                return path
        except Exception:
            pass
    return None


def save_msconvert_path(path):
    try:
        CONFIG_FILE.write_text(str(path), encoding="utf-8")
    except Exception:
        pass


def is_raw_folder(path: Path) -> bool:
    try:
        return path.is_dir() and path.suffix.lower() == ".raw"
    except Exception:
        return False


def find_raw_folders_in(path: Path):
    """
    Given a path, return a list of .raw folders:
    - If path itself is a .raw folder, return [path].
    - Otherwise, if path is a directory, scan its immediate children for .raw folders.
    - Otherwise, return [].
    """
    path = Path(path)
    if not path.exists():
        return []
    if is_raw_folder(path):
        return [path]
    if path.is_dir():
        found = [child for child in path.iterdir() if is_raw_folder(child)]
        return found
    return []


def convert_raw_to_mzml(raw_folders, msconvert_path, output_dir, log_callback=None):
    """
    Runs msconvert.exe on each .raw folder with cyclic IM optimizations.
    Returns list of resulting mzML paths.
        """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mzml_files = []

    for raw_folder in raw_folders:
        raw_folder = Path(raw_folder)
        if log_callback:
            log_callback(f"Converting: {raw_folder.name}")

        cmd = [
            msconvert_path,
            str(raw_folder),
            "-o", str(output_dir),
            "--mzML",
            "--zlib",
            "--simAsSpectra",           # Include SIM functions
            "--srmAsSpectra",           # Include SRM functions
            # "--combineIonMobilitySpectra",  # Cyclic IM support
            "--64",                     # 64-bit precision
        ]

        if log_callback:
            log_callback(f"  Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=600)

            if result.stdout:
                if log_callback:
                    log_callback(result.stdout.strip())

            if result.returncode != 0:
                if log_callback:
                    log_callback(f"WARNING: msconvert returned code {result.returncode}")
                if result.stderr:
                    if log_callback:
                        log_callback(f"STDERR: {result.stderr}")

        except subprocess.TimeoutExpired:
            if log_callback:
                log_callback(f"ERROR: Conversion timed out for {raw_folder.name}")
            continue
        except Exception as e:
            if log_callback:
                log_callback(f"ERROR converting {raw_folder.name}: {e}")
            continue

        expected_mzml = output_dir / (raw_folder.stem + ".mzML")
        if expected_mzml.exists():
            mzml_files.append(expected_mzml)
            if log_callback:
                log_callback(f"  ✓ Created: {expected_mzml.name}")

            # Verify all functions are present
            try:
                funcs = count_functions_in_mzml(expected_mzml)
                if funcs:
                    if log_callback:
                        func_list = sorted(funcs.keys())
                        log_callback(f"  Functions present: {func_list}")
                        log_callback(f"  Total spectra: {sum(funcs.values())}")
            except Exception as e:
                if log_callback:
                    log_callback(f"  (Could not verify functions: {e})")
        else:
            if log_callback:
                log_callback(f"WARNING: expected output not found: {expected_mzml}")

    return mzml_files


def count_functions_in_mzml(mzml_path):
    """
    Count spectra per function in mzML.
    Returns dict: {function_number: spectrum_count}
    """
    try:
        exp = MSExperiment()
        MzMLFile().load(str(mzml_path), exp)

        pattern = re.compile(r"function=(\d+)")
        funcs = {}

        for spec in exp:
            nid = spec.getNativeID()
            m = pattern.search(nid)
            if m:
                f = int(m.group(1))
                funcs[f] = funcs.get(f, 0) + 1

        return funcs
    except Exception as e:
        # Silently fail - pyopenms may not be available
        return {}

def process_tetramer(tetramer_name, mz_range_min, mz_range_max, mzml_path,
                     function_to_voltage, dt_calibration=None,
                     log_callback=None):
    """
    dt_calibration: None  -> use drift times stored in the mzML file
                    (dt_start, dt_interval) -> compute t_drift = dt_start + bin * dt_interval
    """
    exp = MSExperiment()
    MzMLFile().load(str(mzml_path), exp)

    pattern = re.compile(r"function=(\d+)")

    # Force key/value types so 10 and 10.0 can never split into two columns.
    function_to_voltage = {int(f): float(v) for f, v in function_to_voltage.items()}

    half_scan_times = set()
    values = {}
    functions_seen = set()

    for spec in exp:
        # Remove this if there are issues with Linear IMS; level2 corresponds to some data in Cyclic IMS
        # if spec.getMSLevel() != 1:
        #     continue
        mzs, intensities = spec.get_peaks()
        # The following code is dropped to have 200 bins for accurate drift time "calibration"
        # if mzs.size == 0:
        #     continue
        m = pattern.search(spec.getNativeID())
        if not m:
            continue

        func_num = int(m.group(1))
        functions_seen.add(func_num)

        voltage = function_to_voltage.get(func_num)
        if voltage is None:
            continue

        drift_time = float(spec.getDriftTime())
        half_scan_times.add(drift_time)

        mask = (mzs >= mz_range_min) & (mzs <= mz_range_max)
        if np.any(mask):
            key = (voltage, drift_time)
            values[key] = values.get(key, 0.0) + float(np.sum(intensities[mask]))

    sorted_dts = sorted(half_scan_times)

    if log_callback:
        mapped = sorted(function_to_voltage.keys())
        present = sorted(functions_seen)
        log_callback(f"    {Path(mzml_path).name}: functions in file = {present}")
        log_callback(f"    {Path(mzml_path).name}: functions mapped  = {mapped}")
        missing = [f for f in mapped if f not in functions_seen]
        if missing:
            log_callback(f"    WARNING: mapped functions not found in file: {missing}")
        log_callback(f"    {Path(mzml_path).name}: {len(sorted_dts)} distinct drift bins, "
                     f"raw range {sorted_dts[0] if sorted_dts else 'n/a'} "
                     f"-> {sorted_dts[-1] if sorted_dts else 'n/a'}")

    # --- Build the drift-time index ---
    if dt_calibration is None:
        # Round to kill float noise, then dedupe.
        dt_lookup = {dt: round(dt, 6) for dt in sorted_dts}
    else:
        dt_start, dt_interval = dt_calibration
        if sorted_dts:
            # Infer the true bin spacing from the file rather than assuming
            # the observed times are contiguous.
            if len(sorted_dts) > 1:
                diffs = np.diff(sorted_dts)
                raw_spacing = float(np.median(diffs))
            else:
                raw_spacing = 1.0
            dt_first = sorted_dts[0]
            dt_lookup = {}
            for dt in sorted_dts:
                bin_idx = int(round((dt - dt_first) / raw_spacing)) if raw_spacing else 0
                dt_lookup[dt] = round(dt_start + bin_idx * dt_interval, 6)
        else:
            dt_lookup = {}

    index_values = sorted(set(dt_lookup.values()))

    if log_callback and len(index_values) != len(sorted_dts):
        log_callback(f"    WARNING: {len(sorted_dts)} raw drift times collapsed to "
                     f"{len(index_values)} index values — check the calibration interval.")

    columns = sorted(set(function_to_voltage.values()))

    df = pd.DataFrame(0.0, index=index_values, columns=columns)
    for (v, dt), val in values.items():
        df.at[dt_lookup[dt], v] = val
    df.index.name = "drift_time"
    return df

# ============================================================
# USER INTERFACE
# ============================================================

class CIUApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CIU Data Processor v1.7 - By Ismayil Safarli")
        self.geometry("800x800")

        # ---- State ----
        self.raw_folders = []          # list of Path objects (each a .raw folder)
        self.msconvert_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.same_output_var = tk.BooleanVar(value=True)

        self.species_var = tk.StringVar()
        self.comment_var = tk.StringVar()

        # ---- Drift time calibration  ----
        self.use_custom_dt_var = tk.BooleanVar(value=False)
        self.dt_start_var = tk.StringVar(value="")      # t_drift(0)
        self.dt_interval_var = tk.StringVar(value="")   # interval per bin


        self.num_tetramers_var = tk.IntVar(value=1)
        self.tetramer_blocks = []  # list of dicts holding widgets/state per tetramer

        self.available_mzml_files = []   # list of Path, populated by Convert button
        self.mzml_comboboxes = []         # track comboboxes to refresh their values

        self.log_lines = []

        # Load saved msconvert path if present
        saved_path = load_saved_msconvert_path()
        if saved_path:
            self.msconvert_path_var.set(saved_path)

        self._build_ui()
        self.load_session()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self.save_session()
        self.destroy()


    # --------------------------------------------------------
    # UI construction
    # --------------------------------------------------------
    def _build_ui(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        tab1 = ttk.Frame(notebook)
        tab2 = ttk.Frame(notebook)
        notebook.add(tab1, text="1. Files & Conversion")
        notebook.add(tab2, text="2. Tetramers & Functions")

        self._build_tab1(tab1)
        self._build_tab2(tab2)



        # ---- Log area ----
        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10), side="bottom")

        self.log_text = tk.Text(bottom_frame, height=10, state="disabled", bg="#111", fg="#0f0")
        self.log_text.pack(fill="both", expand=True, side="left")

        scrollbar = ttk.Scrollbar(bottom_frame, command=self.log_text.yview)
        scrollbar.pack(side="left", fill="y")
        self.log_text.config(yscrollcommand=scrollbar.set)

    def _get_dt_calibration(self):
        """
        Returns (dt_start, dt_interval) if the custom-calibration box is ticked,
        otherwise None (meaning: use drift times stored in the mzML file).
        Raises ValueError on malformed input.
        """
        if not self.use_custom_dt_var.get():
            return None

        start_str = self.dt_start_var.get().strip()
        interval_str = self.dt_interval_var.get().strip()

        if not start_str or not interval_str:
            raise ValueError(
                "Custom drift time calibration is enabled but Tdrift(0) and/or "
                "the interval is empty. Fill both fields or untick the box."
            )

        try:
            dt_start = float(start_str)
            dt_interval = float(interval_str)
        except ValueError:
            raise ValueError(
                f"Drift time calibration must be numeric "
                f"(got Tdrift(0)='{start_str}', interval='{interval_str}')."
            )

        if dt_interval <= 0:
            raise ValueError(f"Drift time interval must be > 0 (got {dt_interval}).")
        if dt_start < 0:
            raise ValueError(f"Tdrift(0) must be >= 0 (got {dt_start}).")

        return dt_start, dt_interval

    # ---- Tab 1 ----
    def _build_tab1(self, parent):
        # .RAW folders
        raw_frame = ttk.LabelFrame(parent, text=".RAW Folders")
        raw_frame.pack(fill="x", padx=10, pady=5)

        btn_row = ttk.Frame(raw_frame)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="Browse & Add Folder", command=self.browse_add_raw_folder).pack(side="left", padx=5, pady=5)
        ttk.Button(btn_row, text="Remove Selected", command=self.remove_selected_raw_folder).pack(side="left", padx=5, pady=5)
        ttk.Button(btn_row, text="Clear All", command=self.clear_raw_folders).pack(side="left", padx=5, pady=5)

        self.raw_listbox = tk.Listbox(raw_frame, height=5, selectmode="extended")
        self.raw_listbox.pack(fill="x", padx=5, pady=5)

        ttk.Label(raw_frame, text="Or paste folder paths (one per line). "
                                   "You may paste either direct .raw folder paths, "
                                   "or a parent folder containing multiple .raw folders:").pack(anchor="w", padx=5)
        self.raw_paste_text = tk.Text(raw_frame, height=4)
        self.raw_paste_text.pack(fill="x", padx=5, pady=(0, 5))
        ttk.Button(raw_frame, text="Add Pasted Paths", command=self.add_pasted_raw_paths).pack(padx=5, pady=(0, 5))

        # msconvert
        msc_frame = ttk.LabelFrame(parent, text="msconvert.exe")
        msc_frame.pack(fill="x", padx=10, pady=5)

        mode_row = ttk.Frame(msc_frame)
        mode_row.pack(fill="x", padx=5, pady=5)
        ttk.Button(mode_row, text="Try Auto-Detect", command=self.try_auto_detect_msconvert).pack(side="left", padx=5)
        ttk.Button(mode_row, text="Browse...", command=self.browse_msconvert).pack(side="left", padx=5)

        path_row = ttk.Frame(msc_frame)
        path_row.pack(fill="x", padx=5, pady=(0, 5))
        ttk.Label(path_row, text="Path:").pack(side="left")
        self.msconvert_entry = ttk.Entry(path_row, textvariable=self.msconvert_path_var, width=70)
        self.msconvert_entry.pack(side="left", fill="x", expand=True, padx=5)

        ttk.Label(msc_frame,
                  text="Auto-detect scans common install folders. If it fails, browse manually or paste the path above.\n"
                       "The path used successfully will be remembered for next time.",
                  foreground="gray", justify="left").pack(anchor="w", padx=5, pady=(0, 5))

        # Output folder
        out_frame = ttk.LabelFrame(parent, text="Output Folder (for mzML files)")
        out_frame.pack(fill="x", padx=10, pady=5)

        ttk.Checkbutton(out_frame, text="Same as first .raw folder", variable=self.same_output_var,
                         command=self._update_output_mode).pack(anchor="w", padx=5)

        out_row = ttk.Frame(out_frame)
        out_row.pack(fill="x", padx=5, pady=5)
        self.output_entry = ttk.Entry(out_row, textvariable=self.output_dir_var, width=70)
        self.output_entry.pack(side="left", fill="x", expand=True)
        self.output_browse_btn = ttk.Button(out_row, text="Browse...", command=self.browse_output_dir)
        self.output_browse_btn.pack(side="left", padx=5)

        self._update_output_mode()

        # Convert button + status
        convert_frame = ttk.LabelFrame(parent, text="Conversion")
        convert_frame.pack(fill="x", padx=10, pady=5)

        ttk.Button(convert_frame, text="Convert .RAW -> .mzML", command=self.on_convert).pack(padx=5, pady=5)

        self.mzml_status_label = ttk.Label(convert_frame, text="No mzML files generated yet.", foreground="gray")
        self.mzml_status_label.pack(anchor="w", padx=5, pady=(0, 5))


    def _on_toggle_custom_dt(self):
        if self.use_custom_dt_var.get():
            state = "normal"
            hint = "Custom calibration active: bin indices are remapped using the formula above."
        else:
            state = "disabled"
            hint = "Using drift times stored in the mzML file."
        self.dt_start_entry.config(state=state)
        self.dt_interval_entry.config(state=state)
        self.dt_hint_label.config(text=hint)

    # ---- Tab 2 ----
    def _build_tab2(self, parent):
        top_row = ttk.Frame(parent)
        top_row.pack(fill="x", padx=10, pady=5)

        # METADATA
        meta_frame = ttk.LabelFrame(parent, text="Metadata")
        meta_frame.pack(fill="x", padx=10, pady=5)

        # DRIFT TIME CALIBRATION
        dt_frame = ttk.LabelFrame(parent, text="Drift Time Calibration")
        dt_frame.pack(fill="x", padx=10, pady=5)

        check_row = ttk.Frame(dt_frame)
        check_row.pack(fill="x", padx=5, pady=3)
        ttk.Checkbutton(
            check_row,
            text="Use custom drift time (Tdrift(0)) and interval",
            variable=self.use_custom_dt_var,
            command=self._on_toggle_custom_dt
        ).pack(side="left")

        formula_row = ttk.Frame(dt_frame)
        formula_row.pack(fill="x", padx=5, pady=3)
        ttk.Label(
            formula_row,
            text="Tdrift = Tdrift(0) + bin # × interval      ",
            foreground="gray"
        ).pack(side="left")

        dt_row = ttk.Frame(dt_frame)
        dt_row.pack(fill="x", padx=5, pady=3)

        ttk.Label(dt_row, text="Tdrift(0):").pack(side="left")
        self.dt_start_entry = ttk.Entry(dt_row, textvariable=self.dt_start_var, width=10)
        self.dt_start_entry.pack(side="left", padx=5)

        ttk.Label(dt_row, text="Drift time interval:").pack(side="left")
        self.dt_interval_entry = ttk.Entry(dt_row, textvariable=self.dt_interval_var, width=10)
        self.dt_interval_entry.pack(side="left", padx=5)

        self.dt_hint_label = ttk.Label(dt_frame, text="", foreground="gray")
        self.dt_hint_label.pack(anchor="w", padx=5, pady=(0, 5))

        self._on_toggle_custom_dt()   # set initial enabled/disabled state



        label_row = ttk.Frame(meta_frame)
        label_row.pack(fill="x", padx=5, pady=3)
        ttk.Label(label_row, text= ".CSV file label will be built as follows: DT_[mzml file label]_[Species_name]_[Tetramer #]_[mz_min]_[mz_max]_[Comment]_raw").pack(side="left")

        species_row = ttk.Frame(meta_frame)
        species_row.pack(fill="x", padx=5, pady=3)
        ttk.Label(species_row, text="Species name:").pack(side="left")
        ttk.Entry(species_row, textvariable=self.species_var, width=30).pack(side="left", padx=5)

        comment_row = ttk.Frame(meta_frame)
        comment_row.pack(fill="x", padx=5, pady=3)
        ttk.Label(comment_row, text="Comment:").pack(side="left")
        ttk.Entry(comment_row, textvariable=self.comment_var, width=30).pack(side="left", padx=5)

        # Number of Tetramers
        ttk.Label(top_row, text="Number of tetramers:").pack(side="left")
        num_spin = ttk.Spinbox(top_row, from_=1, to=10, textvariable=self.num_tetramers_var, width=5,
                                command=self.rebuild_tetramer_blocks)
        num_spin.pack(side="left", padx=5)
        ttk.Button(top_row, text="Apply", command=self.rebuild_tetramer_blocks).pack(side="left", padx=5)

                # ---- Bottom: START button  ----
        start_row = ttk.Frame(parent)
        start_row.pack(side="bottom", fill="x", padx=5, pady=5)
        ttk.Button(start_row, text="START", command=self.on_start).pack(padx=5, pady=5)

        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        self.tetramer_container = ttk.Frame(canvas)


        self.tetramer_container.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.tetramer_container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=5)
        scrollbar.pack(side="right", fill="y")

        self.rebuild_tetramer_blocks()



    def _clear_dt_calibration(self):
        self.dt_start_var.set("")
        self.dt_interval_var.set("")



    # --------------------------------------------------------
    # .RAW folder handlers
    # --------------------------------------------------------
    def browse_add_raw_folder(self):
        folder = filedialog.askdirectory(title="Select a .raw folder (or its parent folder)")
        if not folder:
            return
        p = Path(folder)
        matches = find_raw_folders_in(p)
        if not matches:
            messagebox.showerror(
                "No .raw folders found",
                f"'{p}' is not a .raw folder, and no .raw subfolders were found inside it."
            )
            self._log(f"No .raw folders found in: {p}")
            return

        added = 0
        for m in matches:
            if m not in self.raw_folders:
                self.raw_folders.append(m)
                self.raw_listbox.insert("end", str(m))
                added += 1
        self._log(f"Added {added} .raw folder(s) from: {p}")

    def add_pasted_raw_paths(self):
        text = self.raw_paste_text.get("1.0", "end").strip()
        if not text:
            return
        lines = [line.strip().strip('"') for line in text.splitlines() if line.strip()]
        added, skipped = 0, []
        for line in lines:
            p = Path(line)
            matches = find_raw_folders_in(p)
            if not matches:
                skipped.append(line)
                continue
            for m in matches:
                if m not in self.raw_folders:
                    self.raw_folders.append(m)
                    self.raw_listbox.insert("end", str(m))
                    added += 1
        self.raw_paste_text.delete("1.0", "end")
        self._log(f"Added {added} .raw folder(s) from pasted paths.")
        if skipped:
            messagebox.showwarning(
                "Some paths skipped",
                "These paths were not .raw folders and contained no .raw subfolders:\n\n" + "\n".join(skipped)
            )
#Method to save data from the current session
    def _collect_session(self):
        data = {
            "output_dir": self.output_dir_var.get(),
            "same_output": self.same_output_var.get(),
            "species": self.species_var.get(),
            "comment": self.comment_var.get(),
            "dt_start": self.dt_start_var.get(),
            "dt_interval": self.dt_interval_var.get(),
            "use_custom_dt": self.use_custom_dt_var.get(),
            "num_tetramers": self.num_tetramers_var.get(),
            "tetramers": [],
        }
        for tb in self.tetramer_blocks:
            tet = {
                "name": tb["name_var"].get(),
                "num_files": tb["num_files_var"].get(),
                "mz_min": tb["mz_min_var"].get(),
                "mz_max": tb["mz_max_var"].get(),
                "file_blocks": [],
            }
            for fb in tb["file_blocks"]:
                tet["file_blocks"].append({
                    "file": fb["file_var"].get(),
                    "num_functions": fb["num_functions_var"].get(),
                    "func_mode": fb["func_mode_var"].get(),
                    "func_start": fb["func_start_var"].get(),
                    "func_incr": fb["func_incr_var"].get(),
                    "volt_mode": fb["volt_mode_var"].get(),
                    "volt_start": fb["volt_start_var"].get(),
                    "volt_incr": fb["volt_incr_var"].get(),
                    "volt_direction": fb["volt_direction_var"].get(),
                    "rows": [{"func": r["func_var"].get(), "volt": r["volt_var"].get()}
                             for r in fb["rows"]],
                })
            data["tetramers"].append(tet)
        return data

    def save_session(self):
        try:
            SESSION_FILE.write_text(json.dumps(self._collect_session(), indent=2),
                                    encoding="utf-8")
            self._log(f"Parameters saved to {SESSION_FILE.name}")
        except Exception as e:
            self._log(f"WARNING: could not save session: {e}")

    def load_session(self):
        if not SESSION_FILE.exists():
            return
        try:
            data = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            self._log(f"WARNING: could not load session: {e}")
            return

        # 1) Top-level fields first
        self.output_dir_var.set(data.get("output_dir", self.output_dir_var.get()))
        self.same_output_var.set(data.get("same_output", self.same_output_var.get()))
        self.species_var.set(data.get("species", self.species_var.get()))
        self.comment_var.set(data.get("comment", self.comment_var.get()))
        self.num_tetramers_var.set(data.get("num_tetramers", len(data.get("tetramers", []))))
        self.use_custom_dt_var.set(data.get("use_custom_dt", False))
        self.dt_start_var.set(data.get("dt_start", "0.0"))
        self.dt_interval_var.set(data.get("dt_interval", "0.05"))
        self._on_toggle_custom_dt()

        # 2) Build the UI shell for the right number of tetramers
        self.rebuild_tetramer_blocks()

        # 3) Refill each tetramer
        for tb, tet in zip(self.tetramer_blocks, data.get("tetramers", [])):
            tb["name_var"].set(tet.get("name", ""))
            tb["mz_min_var"].set(tet.get("mz_min", tb["mz_min_var"].get()))
            tb["mz_max_var"].set(tet.get("mz_max", tb["mz_max_var"].get()))
            tb["num_files_var"].set(tet.get("num_files", len(tet.get("file_blocks", []))))
            self._rebuild_file_blocks(tb)   # see note below

            for fb, fbd in zip(tb["file_blocks"], tet.get("file_blocks", [])):
                fb["file_var"].set(fbd.get("file", ""))
                fb["func_mode_var"].set(fbd.get("func_mode", "auto"))
                fb["func_start_var"].set(fbd.get("func_start", 1))
                fb["func_incr_var"].set(fbd.get("func_incr", 1))
                fb["volt_mode_var"].set(fbd.get("volt_mode", "auto"))
                fb["volt_start_var"].set(fbd.get("volt_start", 0.0))
                fb["volt_incr_var"].set(fbd.get("volt_incr", 5.0))
                fb["volt_direction_var"].set(fbd.get("volt_direction", "increasing"))
                fb["num_functions_var"].set(fbd.get("num_functions", len(fbd.get("rows", []))))
                self._rebuild_rows(fb)

                # restore exact per-row values (overrides auto-fill)
                for r, rd in zip(fb["rows"], fbd.get("rows", [])):
                    r["func_var"].set(rd.get("func", 0))
                    r["volt_var"].set(rd.get("volt", 0.0))

        self._log(f"Parameters restored from {SESSION_FILE.name}")

        #------------------------------------------------------------------------------------------------------------
    def remove_selected_raw_folder(self):
        selected = list(self.raw_listbox.curselection())
        for index in reversed(selected):
            self.raw_listbox.delete(index)
            del self.raw_folders[index]

    def clear_raw_folders(self):
        self.raw_listbox.delete(0, "end")
        self.raw_folders.clear()

    # --------------------------------------------------------
    # msconvert handlers
    # --------------------------------------------------------
    def try_auto_detect_msconvert(self):
        path = find_msconvert()
        if path:
            self.msconvert_path_var.set(path)
            self._log(f"Auto-detected msconvert: {path}")
        else:
            self._log("Auto-detect failed. Please browse manually or paste the path.")
            messagebox.showwarning("Auto-detect failed",
                                    "Could not find msconvert.exe automatically.\n"
                                    "Please use Browse or paste the path manually.")

    def browse_msconvert(self):
        path = filedialog.askopenfilename(title="Select msconvert.exe", filetypes=[("Executable", "*.exe")])
        if path:
            self.msconvert_path_var.set(path)

    # --------------------------------------------------------
    # Output folder handlers
    # --------------------------------------------------------
    def _update_output_mode(self):
        if self.same_output_var.get():
            self.output_entry.config(state="disabled")
            self.output_browse_btn.config(state="disabled")
        else:
            self.output_entry.config(state="normal")
            self.output_browse_btn.config(state="normal")

    def browse_output_dir(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.output_dir_var.set(folder)

    # --------------------------------------------------------
    # Convert button logic
    # --------------------------------------------------------
    def on_convert(self):
        def run_conversion():
            try:
                if not self.raw_folders:
                    self._log("ERROR: No .raw folders selected.")
                    messagebox.showerror("Error", "Please select at least one .raw folder.")
                    return

                self._log(f"{len(self.raw_folders)} .raw folder(s) queued for conversion.")
                self._log("=" * 70)

                msconvert_path = self.msconvert_path_var.get().strip()
                if not msconvert_path or not os.path.isfile(msconvert_path):
                    messagebox.showerror("Error", "Please set a valid msconvert.exe path (Auto-Detect or Browse).")
                    return

                if self.same_output_var.get():
                    output_dir = self.raw_folders[0].parent
                else:
                    output_dir = self.output_dir_var.get().strip()
                    if not output_dir:
                        messagebox.showerror("Error", "Please select an output folder.")
                        return
                    output_dir = Path(output_dir)

                self._log(f"Output folder: {output_dir}")
                self._log(f"Conversion mode: Cyclic IM optimized (preserves all functions)")
                self._log("=" * 70)

                mzml_files = convert_raw_to_mzml(
                    self.raw_folders, 
                    msconvert_path, 
                    output_dir, 
                    log_callback=self._log
                )

                self._log("=" * 70)
                self._log(f"Conversion complete: {len(mzml_files)} mzML file(s) produced.")
                self._log("=" * 70)

                # Save working msconvert path for future runs
                save_msconvert_path(msconvert_path)

                self.available_mzml_files = mzml_files
                self.mzml_status_label.config(
                    text=f"{len(mzml_files)} mzML file(s) available: " +
                         ", ".join(f.name for f in mzml_files),
                    foreground="green"
                )
                self.refresh_all_mzml_dropdowns()

                messagebox.showinfo(
                    "Conversion complete", 
                    f"{len(mzml_files)} mzML file(s) ready for analysis.\n\n"
                    f"All functions (including cyclic IM, SIM, SRM) are preserved.\n"
                    f"Check the log for function verification details."
                )

            except Exception as e:
                self._log(f"ERROR during conversion: {e}")
                messagebox.showerror("Error", f"Conversion failed: {e}")

        thread = threading.Thread(target=run_conversion, daemon=True)
        thread.start()

    def _run_convert(self):
        try:
            self._log("=== Starting .RAW -> .mzML conversion ===")

            if not self.raw_folders:
                messagebox.showerror("Error", "No .RAW folders selected.")
                return

            self._log(f"{len(self.raw_folders)} .raw folder(s) queued for conversion.")

            msconvert_path = self.msconvert_path_var.get().strip()
            if not msconvert_path or not os.path.isfile(msconvert_path):
                messagebox.showerror("Error", "Please set a valid msconvert.exe path (Auto-Detect or Browse).")
                return

            if self.same_output_var.get():
                output_dir = self.raw_folders[0].parent
            else:
                output_dir = self.output_dir_var.get().strip()
                if not output_dir:
                    messagebox.showerror("Error", "Please select an output folder.")
                    return
                output_dir = Path(output_dir)

            self._log(f"Output folder: {output_dir}")

            mzml_files = convert_raw_to_mzml(self.raw_folders, msconvert_path, output_dir, log_callback=self._log)
            self._log(f"Conversion complete: {len(mzml_files)} mzML file(s) produced.")

            # Save working msconvert path for future runs
            save_msconvert_path(msconvert_path)

            self.available_mzml_files = mzml_files
            self.mzml_status_label.config(
                text=f"{len(mzml_files)} mzML file(s) available: " +
                     ", ".join(f.name for f in mzml_files),
                foreground="green"
            )

            self.refresh_all_mzml_dropdowns()
            messagebox.showinfo("Conversion complete", f"{len(mzml_files)} mzML file(s) ready. "
                                                         f"You can now select them in Tab 2.")

        except Exception as e:
            self._log(f"ERROR during conversion: {e}")
            messagebox.showerror("Error", f"Conversion failed: {e}")

    def refresh_all_mzml_dropdowns(self):
        values = [str(f) for f in self.available_mzml_files]
        alive = []
        for combo in self.mzml_comboboxes:
            try:
                combo.config(values=values)
                alive.append(combo)
            except tk.TclError:
                pass  # widget was destroyed; drop it
        self.mzml_comboboxes = alive

    # --------------------------------------------------------
    # Tetramer / function blocks
    # --------------------------------------------------------
    def rebuild_tetramer_blocks(self):
        for block in self.tetramer_blocks:
            block["frame"].destroy()
        self.tetramer_blocks = []
        self.mzml_comboboxes = []

        n = self.num_tetramers_var.get()
        for i in range(n):
            self._add_tetramer_block(i)

    def _add_tetramer_block(self, index):
        frame = ttk.LabelFrame(self.tetramer_container, text=f"Tetramer {index + 1}")
        frame.pack(fill="x", padx=5, pady=5)

        name_row = ttk.Frame(frame)
        name_row.pack(fill="x", padx=5, pady=3)

        mz_row = ttk.Frame(frame)
        mz_row.pack(fill="x", padx=5, pady=3)

        ttk.Label(mz_row, text="m/z min:").pack(side="left")
        mz_min_var = tk.DoubleVar(value=3715)
        ttk.Entry(mz_row, textvariable=mz_min_var, width=10).pack(side="left", padx=5)

        ttk.Label(mz_row, text="m/z max:").pack(side="left")
        mz_max_var = tk.DoubleVar(value=3750)
        ttk.Entry(mz_row, textvariable=mz_max_var, width=10).pack(side="left", padx=5)

        ttk.Label(name_row, text="Name:").pack(side="left")
        name_var = tk.StringVar(value=f"T{index + 1}")
        ttk.Entry(name_row, textvariable=name_var, width=15).pack(side="left", padx=5)

        num_files_row = ttk.Frame(frame)
        num_files_row.pack(fill="x", padx=5, pady=3)
        ttk.Label(num_files_row, text="Number of mzML files for this tetramer:").pack(side="left")
        num_files_var = tk.IntVar(value=1)

        #Values associated with tetramer/block
        block = {
            "frame": frame,
            "name_var": name_var,
            "num_files_var": num_files_var,
            "mz_min_var": mz_min_var,
            "mz_max_var": mz_max_var,
            "file_blocks_container": None,
            "file_blocks": []
        }

        num_files_spin = ttk.Spinbox(num_files_row, from_=1, to=10, textvariable=num_files_var, width=5,
                                      command=lambda: self._rebuild_file_blocks(block))
        num_files_spin.pack(side="left", padx=5)
        ttk.Button(num_files_row, text="Apply", command=lambda: self._rebuild_file_blocks(block)).pack(side="left", padx=5)

        file_blocks_container = ttk.Frame(frame)
        file_blocks_container.pack(fill="x", padx=5, pady=5)
        block["file_blocks_container"] = file_blocks_container

        self.tetramer_blocks.append(block)
        self._rebuild_file_blocks(block)

    def _rebuild_file_blocks(self, tetramer_block):
        for fb in tetramer_block["file_blocks"]:

            cb = fb.get("combobox")
            if cb in self.mzml_comboboxes:
                self.mzml_comboboxes.remove(cb)

            fb["frame"].destroy()
        tetramer_block["file_blocks"] = []

        n = tetramer_block["num_files_var"].get()
        for i in range(n):
            self._add_file_block(tetramer_block, i)

    def _add_file_block(self, tetramer_block, file_index):
        parent = tetramer_block["file_blocks_container"]
        frame = ttk.LabelFrame(parent, text=f"File {file_index + 1}")
        frame.pack(fill="x", padx=5, pady=5)


        file_row = ttk.Frame(frame)
        file_row.pack(fill="x", padx=5, pady=3)
        ttk.Label(file_row, text="mzML file:").pack(side="left")
        file_var = tk.StringVar()
        combo = ttk.Combobox(file_row, textvariable=file_var,
                              values=[str(f) for f in self.available_mzml_files], width=55)
        combo.pack(side="left", padx=5, fill="x", expand=True)
        ttk.Button(file_row, text="Browse...",
                   command=lambda: self._browse_mzml_for_file(file_var)).pack(side="left", padx=5)

        self.mzml_comboboxes.append(combo)

        # Number of functions
        nfunc_row = ttk.Frame(frame)
        nfunc_row.pack(fill="x", padx=5, pady=3)
        ttk.Label(nfunc_row, text="Number of functions in this file:").pack(side="left")
        num_functions_var = tk.IntVar(value=5)

        file_block = {
            "frame": frame,
            "file_var": file_var,
            "combo": combo,
            "num_functions_var": num_functions_var,
            "func_mode_var": tk.StringVar(value="auto"),
            "volt_mode_var": tk.StringVar(value="auto"),
            "func_start_var": tk.IntVar(value=1),
            "func_incr_var": tk.IntVar(value=1),
            "volt_start_var": tk.DoubleVar(value=0.0),
            "volt_incr_var": tk.DoubleVar(value=5.0),
            "volt_direction_var": tk.StringVar(value="increasing"),
            "rows": [],
            "rows_container": None
        }

        nfunc_spin = ttk.Spinbox(nfunc_row, from_=1, to=50, textvariable=num_functions_var, width=5,
                                  command=lambda: self._rebuild_rows(file_block))
        nfunc_spin.pack(side="left", padx=5)
        ttk.Button(nfunc_row, text="Apply", command=lambda: self._rebuild_rows(file_block)).pack(side="left", padx=5)

        # Function mode row
        mode_row = ttk.Frame(frame)
        mode_row.pack(fill="x", padx=5, pady=3)
        ttk.Label(mode_row, text="Function numbers:").pack(side="left")
        ttk.Radiobutton(mode_row, text="Auto-fill", variable=file_block["func_mode_var"],
                         value="auto").pack(side="left", padx=(5, 0))
        ttk.Radiobutton(mode_row, text="Manual", variable=file_block["func_mode_var"],
                         value="manual").pack(side="left")
        ttk.Label(mode_row, text="   Start:").pack(side="left")
        ttk.Entry(mode_row, textvariable=file_block["func_start_var"], width=6).pack(side="left", padx=2)
        ttk.Label(mode_row, text="Step:").pack(side="left")
        ttk.Entry(mode_row, textvariable=file_block["func_incr_var"], width=6).pack(side="left", padx=2)
        ttk.Button(mode_row, text="Fill",
                   command=lambda: self._auto_fill_functions(file_block)).pack(side="left", padx=5)

        # Voltage mode row
        mode_row2 = ttk.Frame(frame)
        mode_row2.pack(fill="x", padx=5, pady=3)
        ttk.Label(mode_row2, text="Voltages:").pack(side="left")
        ttk.Radiobutton(mode_row2, text="Auto-fill", variable=file_block["volt_mode_var"],
                         value="auto").pack(side="left", padx=(5, 0))
        ttk.Radiobutton(mode_row2, text="Manual", variable=file_block["volt_mode_var"],
                         value="manual").pack(side="left")
        ttk.Label(mode_row2, text="   Start:").pack(side="left")
        ttk.Entry(mode_row2, textvariable=file_block["volt_start_var"], width=6).pack(side="left", padx=2)
        ttk.Label(mode_row2, text="Step size:").pack(side="left")
        ttk.Entry(mode_row2, textvariable=file_block["volt_incr_var"], width=6).pack(side="left", padx=2)

        ttk.Radiobutton(mode_row2, text="Increasing", variable=file_block["volt_direction_var"],
                         value="increasing").pack(side="left", padx=(10, 0))
        ttk.Radiobutton(mode_row2, text="Decreasing", variable=file_block["volt_direction_var"],
                         value="decreasing").pack(side="left")

        ttk.Button(mode_row2, text="Fill",
                   command=lambda: self._auto_fill_voltages(file_block)).pack(side="left", padx=5)

        # Rows container (function -> voltage pairs)
        rows_container = ttk.Frame(frame)
        rows_container.pack(fill="x", padx=5, pady=5)
        file_block["rows_container"] = rows_container

        tetramer_block["file_blocks"].append(file_block)
        self._rebuild_rows(file_block)

    def _browse_mzml_for_file(self, file_var):
        path = filedialog.askopenfilename(title="Select mzML file", filetypes=[("mzML files", "*.mzML")])
        if path:
            file_var.set(path)

    def _rebuild_rows(self, file_block):
        for row in file_block["rows"]:
            row["frame"].destroy()
        file_block["rows"] = []

        n = file_block["num_functions_var"].get()
        for i in range(n):
            self._add_row(file_block, i)

        if file_block["func_mode_var"].get() == "auto":
            self._auto_fill_functions(file_block)
        if file_block["volt_mode_var"].get() == "auto":
            self._auto_fill_voltages(file_block)

    def _add_row(self, file_block, row_index):
        parent = file_block["rows_container"]
        row_frame = ttk.Frame(parent)
        row_frame.pack(fill="x", pady=1)

        ttk.Label(row_frame, text=f"Row {row_index + 1}:  Function#").pack(side="left")
        func_var = tk.IntVar(value=0)
        ttk.Entry(row_frame, textvariable=func_var, width=8).pack(side="left", padx=3)

        ttk.Label(row_frame, text="Voltage (V)").pack(side="left")
        volt_var = tk.DoubleVar(value=0.0)
        ttk.Entry(row_frame, textvariable=volt_var, width=8).pack(side="left", padx=3)

        row = {"frame": row_frame, "func_var": func_var, "volt_var": volt_var}
        file_block["rows"].append(row)

    def _auto_fill_functions(self, file_block):
        if file_block["func_mode_var"].get() != "auto":
            return
        start = file_block["func_start_var"].get()
        incr = file_block["func_incr_var"].get()
        for i, row in enumerate(file_block["rows"]):
            row["func_var"].set(start + i * incr)

    def _auto_fill_voltages(self, file_block):
        if file_block["volt_mode_var"].get() != "auto":
            return
        start = file_block["volt_start_var"].get()
        step = file_block["volt_incr_var"].get()
        direction = file_block["volt_direction_var"].get()
        n = len(file_block["rows"])
        if direction == "increasing":
            volts = [start + i * step for i in range(n)]
        else:
            volts = [start - i * step for i in range(n)]

        if any(v < 0 for v in volts):
            messagebox.showerror(
                "Invalid voltage ramp",
                f"Descending ramp from {start} V in {step} V steps over {n} functions "
                f"reaches {min(volts)} V.\nSet start to at least {(n - 1) * step} V."
            )
            return

        for row, v in zip(file_block["rows"], volts):
            row["volt_var"].set(v)

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------
    def _log(self, message):
        self.log_lines.append(message)
        self.log_text.config(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    # --------------------------------------------------------
    # Start / run pipeline (extraction + analysis, assumes mzML already converted)
    # --------------------------------------------------------
    def on_start(self):
        thread = threading.Thread(target=self._run_pipeline, daemon=True)
        thread.start()

    def _run_pipeline(self):
        try:
            self.save_session()
            self._log("=== Starting analysis pipeline ===")

            # Collect all mzML paths actually selected in Tab 2, regardless of
            # whether they came from Convert or from manual Browse/Combobox selection.
            selected_mzml_paths = []
            for block in self.tetramer_blocks:
                for file_block in block["file_blocks"]:
                    p = file_block["file_var"].get().strip()
                    if p:
                        selected_mzml_paths.append(Path(p))

            if not selected_mzml_paths:
                messagebox.showerror(
                    "Error",
                    "No mzML files selected. Please select mzML files in Tab 2 "
                    "(via the dropdown or 'Browse...' button) for at least one tetramer/file block."
                )
                return

            if self.same_output_var.get():
                output_dir = selected_mzml_paths[0].parent
            else:
                output_dir = self.output_dir_var.get().strip()
                if not output_dir:
                    messagebox.showerror("Error", "Please select an output folder.")
                    return
                output_dir = Path(output_dir)

            self._log(f"Output folder: {output_dir}")

            species = self.species_var.get().strip()
            comment = self.comment_var.get().strip()

            dt_calibration = self._get_dt_calibration()
            if dt_calibration is None:
                self._log("Drift time: using values from mzML file(s).")
            else:
                dt_start, dt_interval = dt_calibration
                self._log(f"Drift time: t_drift = {dt_start} + bin * {dt_interval}")

            all_results = {}

            for block in self.tetramer_blocks:
                tetramer_name = block["name_var"].get().strip()

                mz_range_min = block["mz_min_var"].get()
                mz_range_max = block["mz_max_var"].get()

                self._log(
                    f"=== Processing tetramer: {tetramer_name} "
                    f"(m/z {mz_range_min}-{mz_range_max}) ==="
                )

                tetramer_dfs = []

                for file_block in block["file_blocks"]:
                    mzml_path_str = file_block["file_var"].get().strip()
                    if not mzml_path_str:
                        self._log(f"  Skipping file block: no mzML file selected.")
                        continue
                    mzml_path = Path(mzml_path_str)
                    if not mzml_path.exists():
                        self._log(f"  WARNING: mzML file does not exist: {mzml_path}")
                        continue

                    funcs = [row["func_var"].get() for row in file_block["rows"]]
                    volts = [row["volt_var"].get() for row in file_block["rows"]]

                    if len(set(funcs)) != len(funcs):
                        raise ValueError(f"Duplicate function numbers in {mzml_path.name}")
                    if len(set(volts)) != len(volts):
                        raise ValueError(f"Duplicate voltages in {mzml_path.name}")

                    declared = file_block["volt_direction_var"].get()
                    observed = "increasing" if volts[-1] > volts[0] else "decreasing"
                    if declared != observed:
                        self._log(f"  WARNING: {mzml_path.name} declared {declared} "
                                  f"but voltages run {observed} ({volts[0]} → {volts[-1]} V)")

                    # sorted_pairs = sorted(zip(funcs, volts), key=lambda pair: pair[1])
                    # function_to_voltage = dict(sorted_pairs)
                    # function_to_voltage = dict(zip(funcs, volts))
                    function_to_voltage = {int(f): float(v) for f, v in zip(funcs, volts)}

                    self._log(f"{(function_to_voltage)}")

                    self._log(f"  Processing file: {mzml_path.name} "
                              f"with {len(function_to_voltage)} function(s).")

                    df = process_tetramer(
                        tetramer_name,
                        mz_range_min,
                        mz_range_max,
                        mzml_path,
                        function_to_voltage,
                        dt_calibration=dt_calibration,
                        log_callback=self._log,
                    )
                    tetramer_dfs.append(df)

                if not tetramer_dfs:
                    self._log(f"  No valid data extracted for tetramer {tetramer_name}. Skipping.")
                    continue

                # Combine all file DataFrames for this tetramer (side by side on columns,
                # aligned on drift_time index)
                combined_df = pd.concat(tetramer_dfs, axis=1)
                combined_df = combined_df.fillna(0)
                combined_df = combined_df.sort_index()                    # rows
                combined_df = combined_df.reindex(sorted(combined_df.columns), axis=1)   # columns

                all_results[tetramer_name] = combined_df

                # Build filename
                mz_min_str = str(mz_range_min)
                mz_max_str = str(mz_range_max)
                first_file_stem = Path(file_block["file_var"].get().strip()).stem
                label = f"DT_{first_file_stem}_#{species}_{tetramer_name}_{mz_min_str}_{mz_max_str}_{comment}_raw"
                out_path = Path(output_dir) / f"{label}.csv"

                combined_df.to_csv(out_path)
                self._log(f"  Saved: {out_path}")

            self._log("=== Pipeline finished successfully ===")
            messagebox.showinfo("Done", "Analysis pipeline completed successfully.")

        except Exception as e:
            self._log(f"FATAL ERROR: {e}")
            messagebox.showerror("Error", f"Pipeline failed: {e}")


if __name__ == "__main__":
    app = CIUApp()
    app.mainloop()


# In[ ]:




