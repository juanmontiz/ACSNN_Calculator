# Copyright (c) 2025 Juan Manuel Monti
# SPDX-License-Identifier: MIT

"""
GUI Application for Atomic Cross Section Calculations
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import csv
import pandas as pd
from acsnn.config import MODEL_PATH, EXPERIMENTAL_DATA_PATH, DEFAULT_X_MIN, DEFAULT_X_MAX, DEFAULT_N_POINTS, TRAIN_E_MIN, TRAIN_E_MAX
from acsnn import FlexibleNN, predict_conditional


class AtomicCrossSectionApp:
    def __init__(self, root, model_path='models_folder'):
        self.root = root
        self.model_path = model_path
        self.model = FlexibleNN(os.path.join(self.model_path, 'model_weights.npz'))

        self.theories_data = []
        self.experiments_visible = False
        self._editing_index = None

        self.setup_gui()

    def setup_gui(self):
        self.root.title("Multi-Theory Function Plotter")
        self.root.geometry("1400x850")

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=3)
        self.root.rowconfigure(0, weight=1)

        # Scrollable control panel (left column)
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.grid(row=0, column=0, sticky="nswe")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.control_canvas = tk.Canvas(canvas_frame, highlightthickness=0, width=260)
        ctrl_scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.control_canvas.yview)
        self.control_canvas.configure(yscrollcommand=ctrl_scrollbar.set)
        ctrl_scrollbar.pack(side="right", fill="y")
        self.control_canvas.pack(side="left", fill="both", expand=True)

        scrollable_frame = ttk.Frame(self.control_canvas, padding=10)
        self.control_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        # Update scroll region when inner frame resizes
        def _on_frame_configure(event):
            self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
        scrollable_frame.bind("<Configure>", _on_frame_configure)

        # Mousewheel: scroll only when mouse is over the control panel
        def _on_mousewheel(event):
            self.control_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def _on_btn4(event):
            self.control_canvas.yview_scroll(-1, "units")
        def _on_btn5(event):
            self.control_canvas.yview_scroll(1, "units")
        def _bind_wheel(event):
            self.control_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            self.control_canvas.bind_all("<Button-4>", _on_btn4)
            self.control_canvas.bind_all("<Button-5>", _on_btn5)
        def _unbind_wheel(event):
            self.control_canvas.unbind_all("<MouseWheel>")
            self.control_canvas.unbind_all("<Button-4>")
            self.control_canvas.unbind_all("<Button-5>")
        scrollable_frame.bind("<Enter>", _bind_wheel)
        scrollable_frame.bind("<Leave>", _unbind_wheel)

        selection_frame = ttk.Frame(scrollable_frame)
        selection_frame.pack(fill="x")

        ttk.Label(selection_frame, text="Select Theory:").pack(anchor="w", pady=5)
        self.func_var = tk.StringVar()
        self.func_combo = ttk.Combobox(selection_frame, textvariable=self.func_var, state="readonly")
        self.func_combo['values'] = ("CDW-EIS", "CTMC", "Semiempiric_1985Rudd")
        self.func_combo.current(0)
        self.func_combo.pack(fill="x", pady=5)

        self.Zp_var = tk.StringVar()
        self.Zp_var.set('1')
        label_Zp = ttk.Label(selection_frame, text="Projectile Charge:")
        label_Zp.pack(anchor="w", pady=5)
        spin_Zp = ttk.Spinbox(selection_frame, from_=1, to=9, increment=1, textvariable=self.Zp_var, state="readonly")
        spin_Zp.pack(fill="x", pady=5)

        self.Zt_var = tk.StringVar()
        self.Zt_var.set('1')
        label_Zt = ttk.Label(selection_frame, text="Target atomic number:")
        label_Zt.pack(anchor="w", pady=5)
        spin_Zt = ttk.Spinbox(selection_frame, from_=1, to=1, increment=1, textvariable=self.Zt_var, state="readonly")
        spin_Zt.pack(fill="x", pady=5)

        # Visual style controls (placed before Add Theory for easy tweaking)
        ttk.Label(selection_frame, text="Line Type:").pack(anchor="w", pady=5)
        self.linetype_var = tk.StringVar(value='solid')
        linetype_combo = ttk.Combobox(selection_frame, textvariable=self.linetype_var, state="readonly")
        linetype_combo['values'] = ("solid", "dashed", "dotted", "dashdot")
        linetype_combo.pack(fill="x", pady=5)

        ttk.Label(selection_frame, text="Color:").pack(anchor="w", pady=5)
        self.color_var = tk.StringVar(value='tab:blue')
        color_combo = ttk.Combobox(selection_frame, textvariable=self.color_var, state="readonly")
        color_combo['values'] = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan")
        color_combo.pack(fill="x", pady=5)

        ttk.Label(selection_frame, text="Symbol/Marker:").pack(anchor="w", pady=5)
        self.marker_var = tk.StringVar(value='none')
        marker_combo = ttk.Combobox(selection_frame, textvariable=self.marker_var, state="readonly")
        marker_combo['values'] = ("none", "o", ".", "s", "^", "v", "D", "x", "+", "*", "p")
        marker_combo.pack(fill="x", pady=5)

        self.add_theory_btn = tk.Button(
            selection_frame,
            text="Add Theory",
            command=self.add_theory,
            bg="#66bb6a", fg="white", activebackground="#81c784",
            activeforeground="white", relief=tk.RAISED, bd=1,
            font=("TkDefaultFont", 10, "bold")
        )
        self.add_theory_btn.pack(fill="x", pady=5)

        tk.Button(
            selection_frame,
            text="Clear All Theories",
            command=self.clear_theories,
            bg="#ef5350", fg="white", activebackground="#e57373",
            activeforeground="white", relief=tk.RAISED, bd=1,
            font=("TkDefaultFont", 10, "bold")
        ).pack(fill="x", pady=5)

        self.exp_button = tk.Button(
            selection_frame,
            text="Show experimental data",
            command=self.handle_plot_experiments,
            bg="#42a5f5", fg="white", activebackground="#64b5f6",
            activeforeground="white", relief=tk.RAISED, bd=1,
            font=("TkDefaultFont", 10, "bold")
        )
        self.exp_button.pack(fill="x", pady=5)

        ttk.Separator(selection_frame, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(selection_frame, text="Active Theories:").pack(anchor="w", pady=2)
        list_frame = ttk.Frame(selection_frame)
        list_frame.pack(fill="both", expand=True, pady=2)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        self.theory_listbox = tk.Listbox(
            list_frame, height=6, selectmode=tk.SINGLE,
            yscrollcommand=scrollbar.set, exportselection=False
        )
        scrollbar.config(command=self.theory_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.theory_listbox.pack(side="left", fill="both", expand=True)

        self.edit_theory_btn = tk.Button(
            selection_frame,
            text="Edit Selected",
            command=self.edit_selected_theory,
            bg="#b0a090", fg="white", activebackground="#b0a090",
            activeforeground="white", disabledforeground="white",
            relief=tk.RAISED, bd=1,
            font=("TkDefaultFont", 10, "bold"),
            state=tk.DISABLED
        )
        self.edit_theory_btn.pack(fill="x", pady=5)
        self.theory_listbox.bind("<<ListboxSelect>>", self._on_theory_listbox_select)

        tk.Button(
            selection_frame,
            text="Remove Selected",
            command=self.remove_selected_theory,
            bg="#ef5350", fg="white", activebackground="#e57373",
            activeforeground="white", relief=tk.RAISED, bd=1,
            font=("TkDefaultFont", 10, "bold")
        ).pack(fill="x", pady=5)

        self.show_boundaries_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            selection_frame,
            text="Show training range boundaries",
            variable=self.show_boundaries_var,
            command=self.handle_multi_plot_update
        ).pack(anchor="w", pady=5)

        ttk.Separator(scrollable_frame, orient="horizontal").pack(fill="x", pady=10)

        visual_frame = ttk.Frame(scrollable_frame)
        visual_frame.pack(fill="x")

        self.xscale_var = tk.StringVar()
        self.xscale_var.set('log')
        ttk.Label(visual_frame, text="X-axis Scale:").pack(anchor="w", pady=5)
        xscale_combo = ttk.Combobox(visual_frame, textvariable=self.xscale_var, state="readonly")
        xscale_combo['values'] = ("linear", "log")
        xscale_combo.pack(fill="x", pady=5)

        ttk.Label(visual_frame, text="Enter x range (e.g. 10,1000):").pack(anchor="w", pady=5)
        self.xrange_var = tk.StringVar(value="10,10000")
        ttk.Entry(visual_frame, textvariable=self.xrange_var).pack(fill="x", pady=5)

        self.npoints_var = tk.StringVar()
        self.npoints_var.set('300')
        label_npoints = ttk.Label(visual_frame, text="Number of points:")
        label_npoints.pack(anchor="w", pady=5)
        entry_npoints = ttk.Entry(visual_frame, textvariable=self.npoints_var)
        entry_npoints.pack(fill="x", pady=5)

        ttk.Button(
            visual_frame,
            text="Update Plot",
            command=self.handle_multi_plot_update
        ).pack(fill="x", pady=5)

        ttk.Button(
            visual_frame,
            text="Save Plot",
            command=self.handle_save_plot
        ).pack(fill="x", pady=5)

        ttk.Button(
            visual_frame,
            text="Export to CSV",
            command=self.handle_export_csv
        ).pack(fill="x", pady=5)

        plot_frame = ttk.Frame(self.root)
        plot_frame.grid(row=0, column=1, sticky="nswe")

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.redraw()

        def on_closing_window():
            self.root.destroy()
            sys.exit()

        self.root.protocol("WM_DELETE_WINDOW", on_closing_window)

    # ------------------------------------------------------------------
    # Central redraw
    # ------------------------------------------------------------------

    def _window_title(self):
        """Build dynamic window title."""
        if not self.theories_data:
            return "ACSNN Calculator — Multi-Theory Plot"
        names = sorted(set(t['func'] for t in self.theories_data))
        zps = sorted(set(t['Zp'] for t in self.theories_data))
        zps_str = ",".join(f"Zp={z}" for z in zps)
        return f"ACSNN Calculator — {', '.join(names)} ({zps_str})"

    def redraw(self):
        """Redraw everything: theories, experiments, training boundaries, axis settings."""
        self.ax.clear()
        self.root.title(self._window_title())

        if not self.theories_data:
            self.ax.text(0.5, 0.5, 'No theories added. Add theories using the "Add Theory" button.',
                         horizontalalignment='center', verticalalignment='center',
                         transform=self.ax.transAxes, fontsize=12)
            self.ax.set_title("Multi-Theory Plot")
            self.ax.set_xlabel("Projectile Energy (keV/u)")
            self.ax.set_ylabel(r"Cross Section (cm$^2$)")
            self.canvas.draw()
            return

        # Parse current UI state
        try:
            x_min, x_max = map(float, self.xrange_var.get().split(","))
        except ValueError:
            messagebox.showerror("Invalid Input", "x-range must be two comma-separated numbers (e.g. 10,10000).")
            return
        if x_min <= 0 or x_max <= 0 or x_min >= x_max:
            messagebox.showerror("Invalid Input", "x-range requires x_min < x_max and both > 0.")
            return
        npoints = int(self.npoints_var.get())
        if npoints <= 0:
            messagebox.showerror("Invalid Input", "Number of points must be positive.")
            return
        xscale = self.xscale_var.get()

        # Draw theories
        self._draw_theories(x_min, x_max, npoints, xscale)

        # Draw training range boundaries
        if self.show_boundaries_var.get():
            self._draw_training_boundaries(x_min, x_max)

        # Draw experiments over theories and boundaries
        if self.experiments_visible:
            self._draw_experiments(x_min, x_max, xscale)

        # Apply axes settings
        self.ax.set_xscale(xscale)
        self.ax.set_xlim(x_min, x_max)
        self.ax.legend(loc='best', fontsize='small')
        self.ax.set_title(self._plot_title())
        self.ax.set_xlabel("Projectile Energy (keV/u)")
        self.ax.set_ylabel(r"Cross Section (cm$^2$)")
        self.ax.minorticks_on()
        self.ax.tick_params(which='minor', direction='in', bottom=True, top=True, left=True, right=True)
        self.ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        self.canvas.draw()

    def _plot_title(self):
        """Build a dynamic plot title from active theories."""
        if not self.theories_data:
            return "Multi-Theory Plot"
        names = sorted(set(t['func'] for t in self.theories_data))
        zps = sorted(set(t['Zp'] for t in self.theories_data))
        zps_str = ", ".join(f"$Z_P$={z}" for z in zps)
        return f"{', '.join(names)} — {zps_str}, $Z_T$=1 (hydrogen)"

    def _draw_theories(self, x_min, x_max, npoints, xscale):
        """Draw each theory clipped to its training range, masking NaN gaps."""
        for theory_info in self.theories_data:
            theory = theory_info['func']
            Zp = theory_info['Zp']
            Zt = theory_info['Zt']

            # Clip to training range
            tr_min = TRAIN_E_MIN[theory]
            tr_max = TRAIN_E_MAX[theory]
            c_x_min = max(x_min, tr_min)
            c_x_max = min(x_max, tr_max)

            if c_x_min >= c_x_max:
                # No overlap with training range — skip
                continue

            if xscale == 'log':
                x_vals = np.logspace(np.log10(c_x_min), np.log10(c_x_max), npoints)
            else:
                x_vals = np.linspace(c_x_min, c_x_max, npoints)

            if theory in ["CDW-EIS", "CTMC", "Semiempiric_1985Rudd"]:
                y = predict_conditional(self.model, theory, Zp, Zt, np.log10(x_vals))
            else:
                messagebox.showerror("Wrong Model", f"Unknown theory: {theory}")
                continue

            # Mask NaN for clean gaps
            y_flat = y.flatten()
            mask = ~np.isnan(y_flat)
            if not mask.any():
                continue

            label = f"{theory}, $Z_P$={Zp}, $Z_T$=1 (hydrogen)"
            ls = theory_info['linetype']
            color = theory_info['color']
            marker = theory_info['marker']

            plot_kw = dict(linestyle=ls, color=color, label=label, linewidth=1.5)
            if marker != 'none':
                plot_kw['marker'] = marker
                plot_kw['markersize'] = 4

            if xscale == 'log':
                self.ax.loglog(x_vals[mask], y_flat[mask], **plot_kw)
            else:
                self.ax.plot(x_vals[mask], y_flat[mask], **plot_kw)

    def _draw_training_boundaries(self, x_min, x_max):
        """Draw vertical dotted lines at each theory's training range edges."""
        for theory_info in self.theories_data:
            theory = theory_info['func']
            tr_min = TRAIN_E_MIN[theory]
            tr_max = TRAIN_E_MAX[theory]
            color = theory_info['color']
            if x_min < tr_min < x_max:
                self.ax.axvline(tr_min, color=color, linestyle=':', alpha=0.4, linewidth=1)
            if x_min < tr_max < x_max:
                self.ax.axvline(tr_max, color=color, linestyle=':', alpha=0.4, linewidth=1)

    def _draw_experiments(self, x_min, x_max, xscale):
        """Overlay experimental data points for each (Zp, Zt) combination."""
        plotted_combinations = set()
        markers = ['o', 's', '^', 'v', 'D', 'x', '+', '*', 'p', 'h']
        marker_idx = 0

        for theory_info in self.theories_data:
            Zp = theory_info['Zp']
            Zt = theory_info['Zt']
            combination = (Zp, Zt)

            if combination in plotted_combinations:
                continue
            plotted_combinations.add(combination)

            exp_data = self.load_experimental_data(Zp, Zt)
            if exp_data.empty:
                continue

            for source in exp_data['Theory_ID'].unique():
                source_data = exp_data[exp_data['Theory_ID'] == source]
                energies = source_data['E_proj[keV/u]'].values
                cross_sections = source_data['Cross_section[cm2]'].values
                uncertainties = source_data['Uncertainty[%]'].values

                # Filter to visible x range
                in_range = (energies >= x_min) & (energies <= x_max)
                if not in_range.any():
                    continue

                energies = energies[in_range]
                cross_sections = cross_sections[in_range]
                uncertainties = uncertainties[in_range]
                error_bars = (uncertainties / 100.0) * cross_sections

                m = markers[marker_idx % len(markers)]
                marker_idx += 1

                if source.startswith("Exp_"):
                    exp_details = source[4:]
                else:
                    exp_details = source

                exp_label = f'Exp: {exp_details}, $Z_P$={Zp}, $Z_T$=1 (hydrogen)'
                self.ax.errorbar(
                    energies, cross_sections, yerr=error_bars,
                    fmt=m, color='dimgray', capsize=5, capthick=1,
                    label=exp_label, markersize=5, linewidth=1
                )

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def handle_save_plot(self):
        """Save the current plot to a file."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("EPS files", "*.eps"),
                ("SVG files", "*.svg"),
                ("All files", "*.*")
            ],
            title="Save Plot As"
        )

        if file_path:
            try:
                self.fig.savefig(file_path, bbox_inches='tight')
                messagebox.showinfo("Success", f"Plot saved successfully to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plot: {str(e)}")

    def handle_export_csv(self):
        """Export multiple theory data to a single CSV file with NaN handled."""
        if not self.theories_data:
            messagebox.showwarning("No Data", "No theories added. Please add theories first.")
            return

        try:
            x_min, x_max = map(float, self.xrange_var.get().split(","))
        except ValueError:
            messagebox.showerror("Invalid Input", "x-range must be two comma-separated numbers.")
            return
        npoints = int(self.npoints_var.get())

        if self.xscale_var.get() == 'log':
            common_x = np.logspace(np.log10(x_min), np.log10(x_max), npoints)
        else:
            common_x = np.linspace(x_min, x_max, npoints)

        all_y_data = {}

        for theory_info in self.theories_data:
            func = theory_info['func']
            Zp = theory_info['Zp']
            Zt = theory_info['Zt']

            if func in ["CDW-EIS", "CTMC", "Semiempiric_1985Rudd"]:
                y = predict_conditional(self.model, func, Zp, Zt, np.log10(common_x))
                all_y_data[f"TCS_{func}"] = y.flatten() if y.ndim > 1 else y
            else:
                all_y_data[f"TCS_{func}"] = np.full_like(common_x, np.nan)
                messagebox.showerror("Wrong Model")

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ],
            title="Export Multiple Theories to CSV"
        )

        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)

                    # Comment row documenting training ranges
                    range_notes = "; ".join(
                        f"{t['func']}: [{TRAIN_E_MIN[t['func']]}, {TRAIN_E_MAX[t['func']]}] keV/u"
                        for t in self.theories_data
                    )
                    writer.writerow([f"# Training ranges: {range_notes}"])
                    writer.writerow([f"# NaN = outside training range (extrapolation unreliable)"])

                    headers = ["E[keV/u]"] + [f"TCS {t['func']}" for t in self.theories_data]
                    writer.writerow(headers)

                    for i in range(len(common_x)):
                        row = [float(common_x[i])]
                        for theory_info in self.theories_data:
                            func = theory_info['func']
                            y_val = all_y_data[f"TCS_{func}"][i]
                            y_val = float(y_val) if hasattr(y_val, 'item') else y_val
                            row.append("" if np.isnan(y_val) else y_val)
                        writer.writerow(row)

                messagebox.showinfo("Success", f"Multiple theories exported successfully to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export CSV: {str(e)}")

    def load_experimental_data(self, Zp, Zt):
        """Load and filter experimental data for the given Zp and Zt values."""
        exp_file_path = os.path.join(EXPERIMENTAL_DATA_PATH, 'exp_TCS_database_single_ionization_H.csv')
        try:
            df = pd.read_csv(exp_file_path, encoding='utf-8')

            filtered_data = df[(df['Z_ion'] == Zp) & (df['Z_target'] == Zt)]

            return filtered_data
        except FileNotFoundError:
            messagebox.showerror("Error", f"Experimental data file not found at {exp_file_path}")
            return pd.DataFrame()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load experimental data: {str(e)}")
            return pd.DataFrame()

    def _on_theory_listbox_select(self, event=None):
        """Enable/disable Edit Selected based on listbox selection."""
        if self.theory_listbox.curselection():
            self.edit_theory_btn.config(state=tk.NORMAL, bg="#ffa726", activebackground="#ffb74d")
        else:
            self.edit_theory_btn.config(state=tk.DISABLED, bg="#b0a090", activebackground="#b0a090")

    def _update_theory_listbox(self):
        """Refresh the theory listbox to match self.theories_data."""
        self.theory_listbox.delete(0, tk.END)
        for t in self.theories_data:
            self.theory_listbox.insert(tk.END, f"{t['func']}  Zp={t['Zp']}  [{t['color']}]")
        self._on_theory_listbox_select()

    def handle_plot_experiments(self):
        """Toggle experimental data overlay on/off."""
        if not self.theories_data:
            messagebox.showwarning("No Data", "Please add theories first before plotting experiments.")
            return
        self.experiments_visible = not self.experiments_visible
        if self.experiments_visible:
            self.exp_button.config(
                text="Hide experimental data",
                bg="#ffa726", activebackground="#ffb74d"
            )
        else:
            self.exp_button.config(
                text="Show experimental data",
                bg="#42a5f5", activebackground="#64b5f6"
            )
        self.redraw()

    def add_theory(self):
        """Add or update a theory depending on edit mode."""
        try:
            func = self.func_var.get()
            Zp = int(self.Zp_var.get())
            Zt = int(self.Zt_var.get())

            # Duplicate guard — skip entry being edited
            for i, existing in enumerate(self.theories_data):
                if self._editing_index is not None and i == self._editing_index:
                    continue
                if existing['func'] == func and existing['Zp'] == Zp:
                    messagebox.showwarning(
                        "Duplicate Theory",
                        f"{func} with Zp={Zp} is already plotted."
                    )
                    return

            linetype = self.linetype_var.get()
            color = self.color_var.get()
            marker = self.marker_var.get()

            if self._editing_index is not None:
                # --- Update existing theory ---
                theory_info = self.theories_data[self._editing_index]
                theory_info.update({
                    'func': func,
                    'Zp': Zp,
                    'Zt': Zt,
                    'linetype': linetype,
                    'color': color,
                    'marker': marker,
                })
                self._exit_edit_mode()
                self._update_theory_listbox()
                self.redraw()
                messagebox.showinfo("Updated", f"Theory '{func}' updated!")
            else:
                # --- Add new theory ---
                theory_info = {
                    'func': func,
                    'Zp': Zp,
                    'Zt': Zt,
                    'linetype': linetype,
                    'color': color,
                    'marker': marker,
                }
                self.theories_data.append(theory_info)
                self._update_theory_listbox()
                self.redraw()
                messagebox.showinfo("Success", f"Theory '{func}' added to the plot!")

        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please enter valid values: {e}")

    def edit_selected_theory(self):
        """Load selected theory's line properties into comboboxes for editing."""
        selection = self.theory_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Select a theory from the list to edit.")
            return

        idx = selection[0]

        # Toggle off if already editing same selection
        if self._editing_index == idx:
            self._exit_edit_mode()
            return

        theory = self.theories_data[idx]
        self.linetype_var.set(theory['linetype'])
        self.color_var.set(theory['color'])
        self.marker_var.set(theory['marker'])

        self._editing_index = idx
        self.add_theory_btn.config(
            text="Update Theory",
            bg="#7b1fa2", activebackground="#9c27b0"
        )

    def _exit_edit_mode(self):
        """Reset edit mode and restore Add Theory button."""
        self._editing_index = None
        self.add_theory_btn.config(
            text="Add Theory",
            bg="#66bb6a", activebackground="#81c784"
        )

    def remove_selected_theory(self):
        """Remove the theory selected in the listbox."""
        selection = self.theory_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Select a theory from the list to remove.")
            return
        idx = selection[0]
        if self._editing_index is not None:
            self._exit_edit_mode()
        removed = self.theories_data.pop(idx)
        self._update_theory_listbox()
        if not self.theories_data:
            self.experiments_visible = False
        self.redraw()
        messagebox.showinfo("Removed", f"'{removed['func']}' removed from plot.")

    def clear_theories(self):
        """Remove all theories and reset state."""
        if self._editing_index is not None:
            self._exit_edit_mode()
        self.theories_data.clear()
        self.experiments_visible = False
        self.exp_button.config(text="Show experimental data", bg="#42a5f5", activebackground="#64b5f6")
        self._update_theory_listbox()
        self.redraw()
        messagebox.showinfo("Cleared", "All theories have been cleared.")

    def handle_multi_plot_update(self):
        """Re-plot with current widget settings (x-range, npoints, xscale, boundaries)."""
        self.redraw()


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Atomic Cross Section Calculator')
    parser.add_argument('--cli', action='store_true', help='Run in command-line mode')
    parser.add_argument('theory', nargs='?', choices=["CDW-EIS", "CTMC", "Semiempiric_1985Rudd"],
                        help='Theory to use for calculation (CLI mode only)')
    parser.add_argument('Zp', nargs='?', type=int, help='Projectile charge (CLI mode only)')
    parser.add_argument('Zt', nargs='?', type=int, help='Target atomic number (CLI mode only)')
    parser.add_argument('energy_value', nargs='?', type=float,
                        help='Single energy value for calculation (keV/u). If provided, calculates only this value instead of a range. (CLI mode only)')
    parser.add_argument('--x-min', type=float, default=10,
                        help='Minimum x value (CLI mode only, default: 10)')
    parser.add_argument('--x-max', type=float, default=10000,
                        help='Maximum x value (CLI mode only, default: 10000)')
    parser.add_argument('--npoints', type=int, default=300,
                        help='Number of points (CLI mode only, default: 300)')
    parser.add_argument('--output', '-o', help='Output CSV file path (CLI mode only)')
    parser.add_argument('--value-only', action='store_true',
                        help='Output only the numerical value without descriptive text (CLI mode only)')

    args = parser.parse_args()

    if args.cli:
        if not all([args.theory, args.Zp, args.Zt]):
            print("Error: theory, Zp, and Zt are required in CLI mode")
            parser.print_help()
            sys.exit(1)

        from acsnn_cli import run_calculation, run_single_calculation

        try:
            if args.energy_value is not None:
                result = run_single_calculation(
                    theory=args.theory,
                    Zp=args.Zp,
                    Zt=args.Zt,
                    energy_value=args.energy_value
                )

                if args.value_only:
                    print(f"{result}")
                else:
                    print(f"TCS for {args.theory} at E={args.energy_value} keV/u, Zp={args.Zp}, Zt=1 (hydrogen): {result}")
            else:
                results = run_calculation(
                    theory=args.theory,
                    Zp=args.Zp,
                    Zt=args.Zt,
                    x_min=args.x_min,
                    x_max=args.x_max,
                    npoints=args.npoints,
                    output_file=args.output
                )

                if not args.value_only and not args.output:
                    print(f"Calculation completed for {args.theory}, Zp={args.Zp}, Zt=1 (hydrogen)")
                    print("\nFirst 10 results:")
                    print(results.head(10))
                elif not args.output:
                    print(results.to_csv(index=False))

        except Exception as e:
            print(f"Error during calculation: {str(e)}")
            sys.exit(1)
    else:
        root = tk.Tk()
        app = AtomicCrossSectionApp(root)
        root.mainloop()


if __name__ == "__main__":
    main()
