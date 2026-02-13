import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
from io import StringIO
import os
import json
import threading
from tenacity import retry, stop_after_attempt, wait_exponential

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- File Constants ---
DATA_FILE = "exoplanet_data_full.csv"
MOLECULE_DATA_FILE = "planet_molecules.csv"
ATMOSPHERE_JSON_FILE = "atmospheric_signatures.json"

class ExoplanetAnalyzer2D:
    def __init__(self, root):
        self.root = root
        self.root.title("NASA Exoplanet Archive - 2D Research & Atmospheric Analyzer")
        self.root.geometry("1400x900")
        
        # Data storage
        self.df = pd.DataFrame()
        self.mol_df = pd.DataFrame()
        self.signatures = {}
        self.planets_with_spectra = set() 
        self.spectra_cache = {} 
        
        self.load_local_data()
        self.setup_ui()
        
        # Start background fetch for the "available spectra" whitelist
        threading.Thread(target=self.fetch_available_spectra_list, daemon=True).start()

    def load_local_data(self):
        """Loads all required local CSV and JSON files."""
        try:
            if os.path.exists(DATA_FILE):
                # NASA files often have many header comments starting with #
                self.df = pd.read_csv(DATA_FILE, comment='#')
                logging.info(f"Loaded {len(self.df)} planets from {DATA_FILE}")
            
            if os.path.exists(MOLECULE_DATA_FILE):
                self.mol_df = pd.read_csv(MOLECULE_DATA_FILE)
            
            if os.path.exists(ATMOSPHERE_JSON_FILE):
                with open(ATMOSPHERE_JSON_FILE, 'r', encoding='utf-8') as f:
                    self.signatures = json.load(f)
        except Exception as e:
            logging.error(f"Error loading local files: {e}")
            messagebox.showerror("Data Error", "Failed to load local data files. Please ensure CSVs are in the same folder.")

    def fetch_available_spectra_list(self):
        """Identifies which planets have scientific data to avoid redundant API calls."""
        try:
            url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
            query = "select distinct pl_name from transmissionspec"
            params = {'query': query, 'format': 'csv'}
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                names = pd.read_csv(StringIO(response.text))['pl_name'].tolist()
                self.planets_with_spectra = set(names)
                logging.info(f"Whitelisted {len(self.planets_with_spectra)} planets with scientific spectra.")
                self.root.after(0, self.refresh_selector_labels)
        except Exception as e:
            logging.warning(f"Could not fetch spectra whitelist (Offline Mode): {e}")

    def refresh_selector_labels(self):
        """Marks planets in the list that have NASA scientific data."""
        if self.df.empty: return
        all_names = sorted(self.df['pl_name'].unique().tolist())
        display_names = []
        for name in all_names:
            label = f"★ {name}" if name in self.planets_with_spectra else name
            display_names.append(label)
        self.planet_selector['values'] = display_names

    def setup_ui(self):
        # Sidebar for controls
        self.sidebar = ttk.Frame(self.root, padding="15", width=350)
        self.sidebar.pack(side="left", fill="y")
        
        # Notebook for content
        self.main_content = ttk.Notebook(self.root)
        self.main_content.pack(side="right", expand=True, fill="both")

        ttk.Label(self.sidebar, text="Exoplanet Explorer", font=('Helvetica', 14, 'bold')).pack(pady=(0, 20))
        
        ttk.Label(self.sidebar, text="Select a Planet:", font=('Helvetica', 10)).pack(pady=(0, 5))
        self.planet_selector = ttk.Combobox(self.sidebar, state="readonly")
        self.planet_selector.pack(fill="x", pady=5)
        self.planet_selector.bind("<<ComboboxSelected>>", self.update_all_views)
        
        if not self.df.empty:
            self.planet_selector['values'] = sorted(self.df['pl_name'].unique().tolist())

        ttk.Label(self.sidebar, text="Legend:", font=('Helvetica', 9, 'bold')).pack(pady=(20, 5), anchor="w")
        ttk.Label(self.sidebar, text="★ = Scientific Spectra Available", font=('Helvetica', 9, 'italic')).pack(anchor="w")
        ttk.Label(self.sidebar, text="No ★ = Local Research Data Only", font=('Helvetica', 9, 'italic')).pack(anchor="w")

        # Tabs
        self.tab_research = ttk.Frame(self.main_content)
        self.tab_spectrum = ttk.Frame(self.main_content) 
        self.tab_molecules = ttk.Frame(self.main_content)
        
        self.main_content.add(self.tab_research, text="System Research")
        self.main_content.add(self.tab_spectrum, text="Transmission Spectrum")
        self.main_content.add(self.tab_molecules, text="Molecular Abundance")

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
    def fetch_nasa_spectra(self, planet_name):
        """Downloads data only if it exists and isn't cached."""
        if planet_name in self.spectra_cache:
            return self.spectra_cache[planet_name]

        if self.planets_with_spectra and planet_name not in self.planets_with_spectra:
            return None

        try:
            url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
            query = f"select wavelength, flux, flux_err, wavelength_unit, instrument, facility from transmissionspec where pl_name = '{planet_name}'"
            params = {'query': query, 'format': 'csv'}
            response = requests.get(url, params=params, timeout=12)
            if response.status_code == 200 and len(response.text.strip()) > 50:
                data = pd.read_csv(StringIO(response.text))
                self.spectra_cache[planet_name] = data
                return data
            return None
        except Exception as e:
            logging.warning(f"Network fetch failed for {planet_name}: {e}")
            return None

    def update_all_views(self, event=None):
        raw_selection = self.planet_selector.get()
        planet_name = raw_selection.replace("★ ", "")
        
        if not planet_name: return
        
        # Clear all tabs before re-plotting
        for tab in [self.tab_research, self.tab_spectrum, self.tab_molecules]:
            for widget in tab.winfo_children(): widget.destroy()

        # Update everything
        self.plot_research_data(planet_name)
        self.plot_transmission_spectrum(planet_name)
        self.plot_molecules(planet_name)

    def plot_transmission_spectrum(self, planet_name):
        """Handles both real NASA data and theoretical predictions for all planets."""
        parent = self.tab_spectrum
        ttk.Label(parent, text=f"Atmospheric Light Profile: {planet_name}", font=('Helvetica', 14, 'bold')).pack(pady=10)
        
        loading_label = ttk.Label(parent, text="Scanning NASA Archive for Scientific Observations...")
        loading_label.pack()

        def fetch_and_draw():
            spec_data = self.fetch_nasa_spectra(planet_name)
            self.root.after(0, lambda: self.render_spectrum_canvas(parent, planet_name, spec_data, loading_label))

        threading.Thread(target=fetch_and_draw, daemon=True).start()

    def render_spectrum_canvas(self, parent, planet_name, spec_data, placeholder):
        placeholder.destroy()
        
        fig = Figure(figsize=(10, 6), facecolor='#f8f8f8')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#ffffff')

        # Part A: Plot Real Data Points if they exist
        if spec_data is not None:
            for (facility, instrument), group in spec_data.groupby(['facility', 'instrument']):
                label = f"{facility} ({instrument})"
                ax.errorbar(group['wavelength'], group['flux'], yerr=group['flux_err'], 
                            fmt='o', markersize=4, capsize=2, label=label, alpha=0.7)
            
            unit = spec_data['wavelength_unit'].iloc[0]
            ax.set_xlabel(f"Wavelength ({unit})", fontsize=10)
            ax.set_ylabel("Transit Depth (Atmospheric Opacity)", fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
        else:
            # Fallback for planets without scientific data
            ax.text(0.5, 0.5, "No Scientific Spectra Found in NASA Archive.\nVisualizing Theoretical Molecular Bands from local Research.", 
                    ha='center', va='center', color='#555555', transform=ax.transAxes, style='italic')
            ax.set_xlim(0.5, 5.0) # Standard Micron View
            ax.set_ylim(0, 1.2)
            ax.set_xlabel("Wavelength (microns)", fontsize=10)

        # Part B: Overlay Molecular Signatures (for ALL planets)
        if not self.mol_df.empty:
            # Check what chemicals were found in your local planet_molecules.csv
            detected = self.mol_df[self.mol_df['pl_name'] == planet_name]['molecule'].tolist()
            for mol in detected:
                # Get the wavelengths from your signatures JSON
                if mol in self.signatures:
                    color = self.signatures[mol]['color']
                    for wl in self.signatures[mol]['wavelengths']:
                        ax.axvspan(wl-0.03, wl+0.03, color=color, alpha=0.1)
                        ax.text(wl, ax.get_ylim()[1]*0.85, mol.split(' ')[0], color=color, 
                                fontsize=7, rotation=90, ha='right', weight='bold')

        ax.set_title(f"Light Filtration Spectrum: {planet_name}", pad=15)
        ax.grid(True, linestyle=':', alpha=0.5)

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill='both', padx=20, pady=10)

    def plot_research_data(self, planet_name):
        """Displays physical and stellar data for any selected planet."""
        parent = self.tab_research
        # Get data row from exoplanet_data_full.csv
        mask = self.df['pl_name'] == planet_name
        if mask.any():
            planet_info = self.df[mask].iloc[0]
            
            frame = ttk.Frame(parent, padding=30)
            frame.pack(fill="both")
            
            ttk.Label(frame, text=f"Technical Summary: {planet_name}", font=('Helvetica', 16, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 20))
            
            # Helper to handle potential missing numeric data
            def fmt(val, suffix="", dec=2):
                try:
                    if pd.isna(val): return "Data Pending"
                    return f"{float(val):.{dec}f} {suffix}"
                except: return "Unknown"

            details = [
                ("Primary Star:", str(planet_info.get('hostname', 'Unknown'))),
                ("System Distance:", fmt(planet_info.get('sy_dist'), "parsecs")),
                ("Discovery Year:", str(int(planet_info.get('disc_year', 0))) if planet_info.get('disc_year') else "Unknown"),
                ("Orbital Period:", fmt(planet_info.get('pl_orbper'), "days")),
                ("Orbit Radius (a):", fmt(planet_info.get('pl_orbsmax'), "AU", 4)),
                ("Host Star Temperature:", fmt(planet_info.get('st_teff'), "K", 0)),
                ("Discovery Method:", str(planet_info.get('discoverymethod', 'Unknown')))
            ]
            
            for i, (k, v) in enumerate(details):
                ttk.Label(frame, text=k, font=('Helvetica', 11, 'bold')).grid(row=i+1, column=0, sticky="w", pady=5)
                ttk.Label(frame, text=v, font=('Helvetica', 11)).grid(row=i+1, column=1, sticky="w", padx=30)
        else:
            ttk.Label(parent, text="Error: Planet not found in local database.").pack(pady=50)

    def plot_molecules(self, planet_name):
        """Displays chemical composition from local records for any selected planet."""
        parent = self.tab_molecules
        if self.mol_df.empty:
            ttk.Label(parent, text="Error: planet_molecules.csv missing.").pack(pady=20)
            return
        
        planet_mols = self.mol_df[self.mol_df['pl_name'] == planet_name]
        
        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        
        if not planet_mols.empty:
            mols = planet_mols['molecule'].tolist()
            ax.bar(mols, [1]*len(mols), color='#2c3e50', alpha=0.7)
            ax.set_title(f"Detected Atmospheric Components: {planet_name}")
            ax.set_ylim(0, 1.5)
            ax.set_yticks([]) 
            ax.set_ylabel("Presence Confirmed")
        else:
            ax.text(0.5, 0.5, f"No specific molecular detections recorded locally\nfor {planet_name} yet.", 
                    ha='center', va='center', fontsize=12, color='gray')

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill='both', pady=30, padx=30)

if __name__ == "__main__":
    root = tk.Tk()
    # Simple theme adjustment
    style = ttk.Style()
    if 'clam' in style.theme_names():
        style.theme_use('clam')
    app = ExoplanetAnalyzer2D(root)
    root.mainloop()
