# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from scipy.optimize import newton
import requests
import io
import threading
import os

# --- SCIENTIFIC CONSTANTS ---
H_PLANCK = 6.62607015e-34  
C_LIGHT = 299792458         
K_BOLTZMANN = 1.380649e-23 
B_WIEN = 2.897771955e-3    
EARTH_TO_SOLAR = 3.003467e-6   
JUPITER_TO_SOLAR = 9.547919e-4 

class ExoplanetScientificSuite:
    def __init__(self, root):
        self.root = root
        self.root.title("NASA Exoplanet Scientific Suite")
        self.root.geometry("1400x900")
        self.ani = None 
        self.df = pd.DataFrame()
        
        self.load_local_assets()
        self.setup_loading_ui()
        threading.Thread(target=self.fetch_and_organize_data, daemon=True).start()

    def load_local_assets(self):
        mols_path = 'planet_molecules.csv'
        if os.path.exists(mols_path):
            self.mols = pd.read_csv(mols_path, comment='#')
        else:
            # Fallback embedded database
            self.mols = pd.DataFrame([
                {"pl_name": "WASP-39 b", "molecule": "CO2 (Carbon Dioxide)"},
                {"pl_name": "WASP-39 b", "molecule": "H2O (Water)"},
                {"pl_name": "WASP-96 b", "molecule": "H2O (Water)"},
                {"pl_name": "K2-18 b", "molecule": "CH4 (Methane)"},
                {"pl_name": "K2-18 b", "molecule": "CO2 (Carbon Dioxide)"},
            ])

    def setup_loading_ui(self):
        self.loading_frame = ttk.Frame(self.root, padding=50)
        self.loading_frame.pack(expand=True)
        self.status_label = ttk.Label(self.loading_frame, text="Downloading live payload from NASA TAP API...", font=("Arial", 14))
        self.status_label.pack(pady=10)
        self.progress = ttk.Progressbar(self.loading_frame, orient="horizontal", length=400, mode="indeterminate")
        self.progress.pack(pady=10)
        self.progress.start(10)

    def fetch_and_organize_data(self):
        try:
            query = "select pl_name,hostname,pl_orbper,pl_orbsmax,pl_orbeccen,st_teff,st_rad,pl_bmassj,pl_bmasse from ps"
            url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={query}&format=csv"
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            self.root.after(0, lambda: self.status_label.config(text="Data downloaded. Organizing via Pandas..."))
            
            csv_data = io.StringIO(response.text)
            self.df = pd.read_csv(csv_data)
            
            self.df['pl_mass_solar'] = np.nan
            if 'pl_bmassj' in self.df.columns:
                self.df['pl_mass_solar'] = self.df['pl_bmassj'] * JUPITER_TO_SOLAR
            if 'pl_bmasse' in self.df.columns:
                mask = self.df['pl_mass_solar'].isna()
                self.df.loc[mask, 'pl_mass_solar'] = self.df.loc[mask, 'pl_bmasse'] * EARTH_TO_SOLAR

            self.df['pl_orbeccen'] = self.df['pl_orbeccen'].apply(
                lambda x: float(str(x).replace('<', '').replace('>', '').strip()) if pd.notna(x) and str(x).strip() != '' else 0.0
            )
            self.df['pl_orbsmax'] = pd.to_numeric(self.df['pl_orbsmax'], errors='coerce').fillna(1.0)
            self.df['st_teff'] = pd.to_numeric(self.df['st_teff'], errors='coerce')
            self.df['st_rad'] = pd.to_numeric(self.df['st_rad'], errors='coerce')

            self.df = self.df.sort_values('hostname').drop_duplicates(subset=['pl_name'])
            
            self.root.after(0, self.build_main_ui)
            
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(text=f"Connection Error: {e}"))
            self.root.after(0, self.progress.stop)

    def build_main_ui(self):
        self.loading_frame.destroy()

        top_bar = ttk.Frame(self.root, padding=10)
        top_bar.pack(side='top', fill='x')
        
        ttk.Label(top_bar, text="Host Star:").grid(row=0, column=0, padx=5)
        self.star_var = tk.StringVar()
        self.star_combo = ttk.Combobox(top_bar, textvariable=self.star_var, width=25)
        self.star_combo['values'] = sorted(self.df['hostname'].dropna().unique().tolist())
        self.star_combo.grid(row=0, column=1, padx=5)
        self.star_combo.bind("<<ComboboxSelected>>", self.update_planet_list)
        
        ttk.Label(top_bar, text="Planet:").grid(row=0, column=2, padx=5)
        self.planet_var = tk.StringVar()
        self.planet_combo = ttk.Combobox(top_bar, textvariable=self.planet_var, width=25)
        self.planet_combo.grid(row=0, column=3, padx=5)

        ttk.Button(top_bar, text="ANALYSE ATMOSPHERE", command=self.run_analysis).grid(row=0, column=4, padx=15)

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.tab_orbit = ttk.Frame(self.nb)
        self.tab_hr = ttk.Frame(self.nb)
        self.tab_bb = ttk.Frame(self.nb)
        self.tab_atmo = ttk.Frame(self.nb)
        
        self.nb.add(self.tab_orbit, text=" 🪐 Planet Orbit ")
        self.nb.add(self.tab_hr, text=" 🌟 HR Diagram ")
        self.nb.add(self.tab_bb, text=" 🌡️ Black Body Spectrum ")
        self.nb.add(self.tab_atmo, text=" 🔭 Atmospheric Spectra ")

        self.setup_orbit_tab()
        self.setup_hr_tab()
        self.setup_bb_tab()
        self.setup_atmo_tab()

    def setup_orbit_tab(self):
        self.fig_orb, self.ax_orb = plt.subplots(figsize=(6, 6))
        self.canvas_orb = FigureCanvasTkAgg(self.fig_orb, master=self.tab_orbit)
        self.canvas_orb.get_tk_widget().pack(fill='both', expand=True)

    def setup_hr_tab(self):
        self.fig_hr, self.ax_hr = plt.subplots(figsize=(6, 6))
        self.canvas_hr = FigureCanvasTkAgg(self.fig_hr, master=self.tab_hr)
        self.canvas_hr.get_tk_widget().pack(fill='both', expand=True)
        
        valid_stars = self.df.dropna(subset=['st_teff', 'st_rad']).drop_duplicates(subset=['hostname'])
        self.ax_hr.scatter(valid_stars['st_teff'], valid_stars['st_rad'], s=2, color='gray', alpha=0.3, label='All Stars')
        self.ax_hr.set_yscale('log')
        self.ax_hr.invert_xaxis()
        self.ax_hr.set_title("Hertzsprung-Russell Diagram")
        self.ax_hr.set_xlabel("Effective Temperature (K)")
        self.ax_hr.set_ylabel("Radius (Solar Radii)")
        
        self.hr_highlight, = self.ax_hr.plot([], [], 'ro', markersize=10, markeredgecolor='black', label='Selected Star')
        self.ax_hr.legend()

    def setup_bb_tab(self):
        self.fig_bb, self.ax_bb = plt.subplots(figsize=(6, 6))
        self.canvas_bb = FigureCanvasTkAgg(self.fig_bb, master=self.tab_bb)
        self.canvas_bb.get_tk_widget().pack(fill='both', expand=True)

    def setup_atmo_tab(self):
        self.fig_atmo, self.ax_atmo = plt.subplots(figsize=(6, 6))
        self.canvas_atmo = FigureCanvasTkAgg(self.fig_atmo, master=self.tab_atmo)
        self.canvas_atmo.get_tk_widget().pack(fill='both', expand=True)
        self.ax_atmo.text(0.5, 0.5, "Select a planet and click [ RUN ANALYSIS ]", ha='center', va='center', fontsize=12)

    def update_planet_list(self, event=None):
        star = self.star_var.get()
        self.planet_combo['values'] = sorted(self.df[self.df['hostname'] == star]['pl_name'].tolist())
        if self.planet_combo['values']:
            self.planet_combo.current(0)

    def run_analysis(self):
        p_name = self.planet_var.get()
        if not p_name: return
            
        p_data = self.df[self.df['pl_name'] == p_name].iloc[0]
        
        self.draw_orbit(p_data)
        self.update_hr_diagram(p_data)
        self.draw_black_body(p_data)
        
        # This will automatically find and plot the file
        self.auto_plot_spectra(p_name)

        # Molecule Check
        atmo_data = self.mols[self.mols['pl_name'] == p_name]
        if not atmo_data.empty:
            mols_found = ", ".join(atmo_data['molecule'].dropna().tolist())
            messagebox.showinfo("Atmosphere Data", f"SUCCESS: Atmospheric data verified.\n\nDetections: {mols_found}")
        else:
            pass # Removed the annoying error popup so it doesn't interrupt the user experience

    def auto_plot_spectra(self, p_name):
        self.ax_atmo.clear()
        
        # 1. Format the planet name to match NASA file naming conventions
        # Example: "WASP-39 b" becomes "WASP_39_b"
        file_prefix = p_name.replace(' ', '_').replace('-', '_')
        
        found_files = []
        
        # 2. AUTO-SEARCH: Hunt down any .tbl file that starts with this prefix in the entire project folder
        for root_dir, dirs, files in os.walk('.'):
            for file in files:
                if file.startswith(file_prefix) and file.endswith('.tbl'):
                    found_files.append(os.path.join(root_dir, file))
        
        if not found_files:
            self.ax_atmo.text(0.5, 0.5, f"NO LOCAL SPECTRA DATA FOUND FOR:\n{p_name}", 
                              color='red', ha='center', va='center', fontweight='bold', fontsize=12)
            self.ax_atmo.set_title(f"Atmospheric Spectra for {p_name}")
            self.canvas_atmo.draw()
            return

        plotted = False
        
        # 3. ROBUST CUSTOM PARSER: Loop through all matching files and plot them
        for file_path in found_files:
            x_vals, y_vals, err_vals = [], [], []
            
            try:
                # Read the file manually to prevent Pandas formatting crashes
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        # Ignore NASA IPAC headers and comments
                        if not line or line.startswith('\\') or line.startswith('|'):
                            continue
                        
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                x_vals.append(float(parts[0])) # Wavelength
                                y_vals.append(float(parts[1])) # Transit Depth
                                if len(parts) >= 3:
                                    err_vals.append(float(parts[2])) # Margin of Error
                            except ValueError:
                                continue # Skip lines that don't have numbers
                
                # Plot the extracted data
                if x_vals and y_vals:
                    if len(err_vals) == len(x_vals):
                        self.ax_atmo.errorbar(x_vals, y_vals, yerr=err_vals, fmt='o', ms=4, alpha=0.7, capsize=2, label=os.path.basename(file_path))
                    else:
                        self.ax_atmo.plot(x_vals, y_vals, 'o', ms=4, alpha=0.7, label=os.path.basename(file_path))
                    plotted = True
            
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        if plotted:
            self.ax_atmo.set_title(f"Observed Transmission Spectra: {p_name}")
            self.ax_atmo.set_xlabel("Wavelength (microns)")
            self.ax_atmo.set_ylabel("Transit Depth")
            self.ax_atmo.grid(True, linestyle=':', alpha=0.6)
            
            # Only show legend if there are a few files, otherwise it blocks the graph
            if len(found_files) <= 5:
                self.ax_atmo.legend()
        else:
            self.ax_atmo.text(0.5, 0.5, "FILES FOUND BUT COULD NOT EXTRACT DATA", color='red', ha='center', va='center', fontweight='bold')
            
        self.canvas_atmo.draw()

    def draw_orbit(self, p_data):
        a = float(p_data['pl_orbsmax']) if pd.notna(p_data['pl_orbsmax']) else 1.0
        e = float(p_data['pl_orbeccen'])
        
        self.ax_orb.clear()
        self.ax_orb.set_facecolor('#000000')
        self.ax_orb.plot(0, 0, 'yo', markersize=12, label='Host Star') 

        nu_path = np.linspace(0, 2*np.pi, 500)
        r_path = (a * (1 - e**2)) / (1 + e * np.cos(nu_path))
        self.ax_orb.plot(r_path * np.cos(nu_path), r_path * np.sin(nu_path), 'w--', alpha=0.3)

        planet_dot, = self.ax_orb.plot([], [], 'ro', markersize=8, label=p_data['pl_name'])
        
        def update(frame):
            M = (2 * np.pi * frame) / 100 
            E = newton(lambda x: x - e*np.sin(x) - M, M) 
            nu = 2 * np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2)) 
            r = (a * (1 - e**2)) / (1 + e * np.cos(nu)) 
            x, y = r * np.cos(nu), r * np.sin(nu)
            planet_dot.set_data([x], [y])
            self.ax_orb.set_title(f"Distance: {r:.4f} AU", color='white')
            return planet_dot,

        limit = a * 1.5
        self.ax_orb.set_xlim(-limit, limit)
        self.ax_orb.set_ylim(-limit, limit)
        self.ax_orb.set_aspect('equal')
        
        if self.ani: self.ani.event_source.stop()
        self.ani = animation.FuncAnimation(self.fig_orb, update, frames=100, interval=50, blit=False)
        self.canvas_orb.draw()

    def update_hr_diagram(self, p_data):
        T = p_data['st_teff']
        R = p_data['st_rad']
        
        if pd.notna(T) and pd.notna(R):
            self.hr_highlight.set_data([T], [R])
            self.hr_highlight.set_label(f"{p_data['hostname']} (T={T}K, R={R}R☉)")
            self.ax_hr.legend()
        else:
            self.hr_highlight.set_data([], [])
        self.canvas_hr.draw()

    def draw_black_body(self, p_data):
        T = p_data['st_teff']
        self.ax_bb.clear()
        
        if pd.notna(T):
            wav = np.linspace(100e-9, 3000e-9, 1000)
            intensity = (2 * H_PLANCK * C_LIGHT**2) / (wav**5 * (np.exp((H_PLANCK * C_LIGHT) / (wav * K_BOLTZMANN * T)) - 1))
            
            self.ax_bb.plot(wav * 1e9, intensity, color='orange', lw=2)
            self.ax_bb.fill_between(wav * 1e9, intensity, color='orange', alpha=0.3)
            
            self.ax_bb.set_title(f"Black Body Spectrum for {p_data['hostname']} (T = {T} K)")
            self.ax_bb.set_xlabel("Wavelength (nm)")
            self.ax_bb.set_ylabel("Spectral Radiance")
            self.ax_bb.grid(True, linestyle=':', alpha=0.6)
            
            peak_wav = (B_WIEN / T) * 1e9
            self.ax_bb.axvline(peak_wav, color='red', linestyle='--', label=f'Peak Wavelength: {peak_wav:.1f} nm')
            self.ax_bb.legend()
        else:
            self.ax_bb.text(0.5, 0.5, "Temperature data not available.", ha='center', va='center', color='red')
            
        self.canvas_bb.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = ExoplanetScientificSuite(root)
    root.mainloop()