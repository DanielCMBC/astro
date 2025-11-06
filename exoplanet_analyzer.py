import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import newton
import logging
from io import StringIO
import os
import time
import shutil
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from astroquery.mast import Observations
from astropy.io import fits


H_PLANCK = 6.626e-34
C_LIGHT = 3e8
K_BOLTZMANN = 1.381e-23
SOLAR_TEMP = 5778


# These are the files you just uploaded.
PLANET_DATA_FILE = "exoplanets_with_atmo_fields.csv"
ATMOSPHERE_JSON_FILE = "atmospheric_signatures.json"
MOLECULE_DATA_FILE = "planet_molecules.csv"


try:
    with open(ATMOSPHERE_JSON_FILE, 'r') as f:
        ATMOSPHERIC_SIGNATURES = json.load(f)
    logging.info(f"Successfully loaded {len(ATMOSPHERIC_SIGNATURES)} signatures from {ATMOSPHERE_JSON_FILE}")
except Exception as e:
    logging.error(f"FATAL ERROR: Could not read {ATMOSPHERE_JSON_FILE}. {e}")
    messagebox.showerror("File Error", f"Could not read {ATMOSPHERE_JSON_FILE}. App will exit.")
    ATMOSPHERIC_SIGNATURES = {} # Fallback
    exit()



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('exoplanet_analyzer.log', mode='w')] 
)

class ExoplanetAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("NASA Exoplanet Archive Analyzer (Local Version)")
        self.root.geometry("1280x900")
        self.root.minsize(1000, 700)
        self.ani = None
        
     
        self.planetary_data, self.molecule_detections = self.load_local_data()

        if not self.planetary_data.empty:
            self.host_stars = sorted(self.planetary_data['hostname'].dropna().unique())
        else:
            self.host_stars = []
            # Error is handled in the load function
            
        self.setup_ui()

        if self.host_stars:
            self.star_var.set(self.host_stars[0])
            self.update_planets()

    def load_local_data(self) -> (pd.DataFrame, pd.DataFrame):
        """Loads all data from the local CSV files."""
        logging.info(f"Attempting to load data from local files...")
        
        if not os.path.exists(PLANET_DATA_FILE) or not os.path.exists(MOLECULE_DATA_FILE):
            error_msg = f"FATAL ERROR: Data file not found."
            logging.error(error_msg)
            messagebox.showerror("File Not Found", f"Could not find '{PLANET_DATA_FILE}' or '{MOLECULE_DATA_FILE}'.\nPlease make sure they are in the same folder as the script.")
            self.root.destroy()
            return pd.DataFrame(), pd.DataFrame()
        
        try:
            # Load the main planet "phonebook"
            planets_df = pd.read_csv(PLANET_DATA_FILE, comment='#')
            logging.info(f"Successfully loaded {len(planets_df)} records from {PLANET_DATA_FILE}.")

            # Load the molecule detection "join table"
            molecules_df = pd.read_csv(MOLECULE_DATA_FILE, comment='#')
            logging.info(f"Successfully loaded {len(molecules_df)} records from {MOLECULE_DATA_FILE}.")

       
            # Rename columns from the CSV to match what the script expects
            column_rename_map = {
                'st_teff': 'teff_gspphot',
                'st_lum': 'lum_gspphot',
                'st_rad': 'radius_gspphot',
                'sy_bp_rp': 'bp_rp'
            }
            
            cols_to_rename = {k: v for k, v in column_rename_map.items() if k in planets_df.columns}
            planets_df = planets_df.rename(columns=cols_to_rename)

            # Ensure all expected columns exist, adding NaNs if they don't
            expected_cols = ['pl_name', 'hostname', 'pl_orbper', 'pl_orbsmax', 'pl_radj',
                             'pl_bmassj', 'disc_year', 'pl_orbeccen', 'st_mass', 'st_rad', 'st_age',
                             'teff_gspphot', 'st_spectype', 'sy_dist', 'lum_gspphot', 'radius_gspphot', 'bp_rp']
            
            for col in expected_cols:
                if col not in planets_df.columns:
                    logging.warning(f"Column '{col}' not found in CSV. Creating it with NaN values.")
                    planets_df[col] = np.nan
            
            return planets_df, molecules_df

        except Exception as e:
            logging.error(f"Failed to read or process local CSV files: {e}", exc_info=True)
            messagebox.showerror("Data Error", f"Failed to read local data files.\nError: {e}")
            self.root.destroy()
            return pd.DataFrame(), pd.DataFrame()

    def fetch_atmospheric_data(self, planet_name: str) -> pd.DataFrame:
        """
        NEW: Gets molecular composition data from the local 'planet_molecules.csv'.
        This function no longer queries the internet.
        """
        logging.info(f"Fetching local molecular data for {planet_name}...")
        if self.molecule_detections.empty:
            logging.warning("Molecule detections dataframe is empty.")
            return pd.DataFrame()
        
        # Filter the local dataframe for the selected planet
        detections = self.molecule_detections[
            self.molecule_detections['pl_name'].str.lower() == planet_name.lower()
        ]
        
        if detections.empty:
            logging.info(f"No local molecule detections found for {planet_name}.")
            return pd.DataFrame()
            
        # Re-format the dataframe to match the old structure (molecule, abundance)
        # Since we only have *detections*, we'll just plot them with a placeholder abundance of 1
        final_df = detections[['molecule']].copy()
        final_df['abundance'] = 1.0 
        final_df['abundance_err'] = 0
        
        logging.info(f"Found {len(final_df)} local molecule detections for {planet_name}.")
        return final_df

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_spectroscopy_data(self, planet_name: str) -> pd.DataFrame:
        """
        Gets transmission spectroscopy data from MAST archive (Hubble/JWST).
        This function STILL queries the internet, as it's for real-time scientific data.
        """
        formatted_planet_name = planet_name.replace(" ", "")
        logging.info(f"Attempting MAST query for {planet_name} as '{formatted_planet_name}'")
        
        shutil.rmtree('mastDownload', ignore_errors=True)

        try:
            obs_table = Observations.query_criteria(
                target_name=formatted_planet_name,
                obs_collection=["HST", "JWST"],
                dataproduct_type="spectrum",
                calib_level=3 
            )

            if len(obs_table) == 0:
                logging.warning(f"No JWST or HST spectrum data found on MAST for {formatted_planet_name}")
                return pd.DataFrame()
            logging.info(f"Found {len(obs_table)} observations on MAST for {formatted_planet_name}.")

            data_products = Observations.get_product_list(obs_table)
            logging.info(f"Found {len(data_products)} total data products.")
            
            science_products = None
            for suffix in ['x1d.fits', 'x1dints.fits', 's2d.fits']: 
                 current_science_products = Observations.filter_products(
                     data_products,
                     productType="SCIENCE",
                     productSubGroupDescription=suffix.upper()
                 )
                 if len(current_science_products) > 0:
                     science_products = current_science_products
                     logging.info(f"Found {len(science_products)} '{suffix}' files. Proceeding with the first one.")
                     break
            
            if science_products is None or len(science_products) == 0:
                logging.warning(f"Found observations for {formatted_planet_name}, but no suitable FITS products ('x1d', 'x1dints', 's2d').")
                return pd.DataFrame()

            target_product = science_products[0]
            logging.info(f"Attempting to download: {target_product['productFilename']}")
            manifest = Observations.download_products(
                target_product,
                download_dir="." 
            )
            
            if manifest is None or len(manifest) == 0 or 'Local Path' not in manifest.columns or manifest[0]['Status'] != 'COMPLETE':
                 logging.error(f"MAST download failed or incomplete for {target_product['productFilename']}. Status: {manifest[0]['Status'] if manifest is not None and len(manifest)>0 else 'Unknown'}")
                 shutil.rmtree('mastDownload', ignore_errors=True)
                 return pd.DataFrame()
            
            local_file_path = manifest[0]['Local Path']
            logging.info(f"Successfully downloaded to: {local_file_path}")
            
            with fits.open(local_file_path) as hdul:
                data_hdu = None
                hdu_info = []
                for i, hdu in enumerate(hdul):
                    hdu_info.append(f"HDU {i}: Name={hdu.name}, Type={type(hdu).__name__}, Columns={hdu.columns.names if hasattr(hdu, 'columns') else 'N/A'}")
                    if isinstance(hdu, fits.BinTableHDU) and 'WAVELENGTH' in [c.upper() for c in hdu.columns.names]:
                        data_hdu = hdu
                        logging.info(f"Found spectrum data in FITS extension {i} ('{hdu.name}') (BinTableHDU)")
                        break
                
                if data_hdu is None:
                    logging.warning("No BinTableHDU with WAVELENGTH found. Looking for 'SCI' extension...")
                    for i, hdu in enumerate(hdul):
                         if hdu.name == 'SCI':
                             logging.warning(f"Found 'SCI' extension at index {i}. This might be 2D data requiring complex parsing (not fully supported).")
                             break 

                if data_hdu is None:
                    logging.error(f"Could not find a suitable data table (BinTableHDU with WAVELENGTH) in {local_file_path}. HDU structure: {'; '.join(hdu_info)}")
                    shutil.rmtree('mastDownload', ignore_errors=True)
                    return pd.DataFrame()

                data = data_hdu.data
                if data.dtype.byteorder not in ('=', '|'):
                    df = pd.DataFrame(data.byteswap().newbyteorder())
                    logging.debug("Swapped byte order for FITS data.")
                else:
                    df = pd.DataFrame(data)
                logging.info(f"Successfully read data from HDU '{data_hdu.name}'. DataFrame shape: {df.shape}")

            shutil.rmtree('mastDownload', ignore_errors=True)
            logging.info("Cleaned up mastDownload directory.")

            out_df = pd.DataFrame()
            col_map = {
                'wavelength': ['WAVELENGTH', 'wave'],
                'flux': ['FLUX', 'flux'],
                'flux_err': ['FLUX_ERROR', 'FLUX_ERR', 'err', 'error']
            }
            
            df_cols_upper = [str(c).upper() for c in df.columns]
            for standard_col, possible_names in col_map.items():
                found = False
                for name in possible_names:
                    if name.upper() in df_cols_upper:
                        original_col_name = df.columns[df_cols_upper.index(name.upper())]
                        out_df[standard_col] = df[original_col_name]
                        logging.info(f"Mapped FITS column '{original_col_name}' to '{standard_col}'")
                        found = True
                        break
                if not found and standard_col == 'flux_err':
                     logging.warning(f"Could not map '{standard_col}' column. Will set errors to NaN.")
                     out_df['flux_err'] = np.nan
                elif not found:
                     logging.error(f"CRITICAL: Could not map required column '{standard_col}'. Available FITS columns: {list(df.columns)}")
                     return pd.DataFrame()

            unit = 'um' 
            wl_col_name_fits = None
            try:
                wl_cols_upper_fits = [str(c).upper() for c in data_hdu.columns.names]
                for name in col_map['wavelength']:
                     if name.upper() in wl_cols_upper_fits:
                         wl_col_name_fits = data_hdu.columns.names[wl_cols_upper_fits.index(name.upper())]
                         break
                
                if wl_col_name_fits:
                    wl_col_index_fits = list(data_hdu.columns.names).index(wl_col_name_fits)
                    unit_key = f'TUNIT{wl_col_index_fits + 1}'
                    if unit_key in data_hdu.header:
                         unit = data_hdu.header[unit_key]
                         logging.info(f"Detected wavelength unit '{unit}' from FITS header key '{unit_key}'.")
                    else:
                         logging.warning(f"Wavelength unit key '{unit_key}' not found in FITS header. Defaulting to 'um'.")
                else:
                     logging.warning("Could not find original FITS wavelength column name to check TUNIT. Defaulting to 'um'.")

            except Exception as e:
                logging.warning(f"Could not parse wavelength unit from FITS header (TUNIT): {e}. Defaulting to 'um'.")
            
            if unit:
                unit = unit.lower().replace('micron', 'um').strip()
            else: 
                unit = 'um'

            out_df['wavelength_unit'] = unit
            out_df['instrument'] = obs_table[0]['instrument_name']

            original_rows = len(out_df)
            out_df = out_df[out_df['wavelength'] > 0].dropna(subset=['wavelength'])
            filtered_rows = len(out_df)
            if original_rows != filtered_rows:
                 logging.warning(f"Filtered out {original_rows - filtered_rows} rows with invalid wavelength values.")

            if out_df.empty:
                 logging.error("DataFrame is empty after filtering invalid wavelengths.")
                 return pd.DataFrame()

            logging.info(f"Successfully processed FITS data for {planet_name}. Final DataFrame shape: {out_df.shape}")
            return out_df

        except FileNotFoundError:
             logging.error(f"Downloaded FITS file not found. Check download process.")
             return pd.DataFrame()
        except Exception as e:
            logging.error(f"MAST query or FITS processing failed for {planet_name}: {str(e)}", exc_info=True)
            shutil.rmtree('mastDownload', ignore_errors=True)
            return pd.DataFrame()
        
    def setup_ui(self):
        """Initialize main UI components"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        self.create_orbit_tab()
        self.create_spectrum_tab()
        self.create_hr_tab()
        self.create_research_tab()
        self.create_controls()

    def create_controls(self):
        """Create control panel"""
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(control_frame, text="Host Star:").grid(row=0, column=0, padx=5)
        self.star_var = tk.StringVar()
        self.star_combo = ttk.Combobox(control_frame, textvariable=self.star_var,
                                      values=self.host_stars, state='readonly',
                                      width=30)
        self.star_combo.grid(row=0, column=1, padx=5)
        self.star_combo.bind("<<ComboboxSelected>>", self.update_planets)

        ttk.Label(control_frame, text="Planet:").grid(row=0, column=2, padx=5)
        self.planet_var = tk.StringVar()
        self.planet_combo = ttk.Combobox(control_frame, textvariable=self.planet_var,
                                       state='readonly', width=30)
        self.planet_combo.grid(row=0, column=3, padx=5)

        ttk.Button(control_frame, text="Animate Orbit",
                   command=self.animate_orbit).grid(row=0, column=4, padx=5)
        ttk.Button(control_frame, text="Show Atmosphere",
                   command=self.show_atmosphere).grid(row=0, column=5, padx=5)
        
        # Removed "Refresh Data" button as it's no longer needed for main data
        # ttk.Button(control_frame, text="Refresh Data",
        #            command=self.refresh_data).grid(row=0, column=6, padx=5)

    def create_orbit_tab(self):
        """Configure orbit animation tab"""
        self.orbit_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.orbit_tab, text="Orbit Viewer")

        self.orbit_info = tk.Text(self.orbit_tab, height=8, wrap=tk.WORD, state='disabled')
        self.orbit_info.pack(fill='x', padx=10, pady=(10, 0))

        self.fig_orbit = Figure(figsize=(8, 6), facecolor='#0a0a2a')
        self.ax_orbit = self.fig_orbit.add_subplot(111)
        self.ax_orbit.set_facecolor("black")
        self.canvas_orbit = FigureCanvasTkAgg(self.fig_orbit, master=self.orbit_tab)
        self.canvas_orbit.get_tk_widget().pack(expand=True, fill='both', padx=10, pady=10)
        self.ax_orbit.text(0.5, 0.5, "Select a planet and click 'Animate Orbit'", color='white',
                           ha='center', va='center', transform=self.ax_orbit.transAxes)


    def create_spectrum_tab(self):
        """Configure black body spectrum tab"""
        self.spectrum_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.spectrum_tab, text="Stellar Spectrum")

        self.fig_spectrum = Figure(figsize=(8, 6))
        self.ax_spectrum = self.fig_spectrum.add_subplot(111)
        self.canvas_spectrum = FigureCanvasTkAgg(self.fig_spectrum, master=self.spectrum_tab)
        self.canvas_spectrum.get_tk_widget().pack(expand=True, fill='both', padx=10, pady=10)
        self.ax_spectrum.text(0.5, 0.5, "Select a star to view its spectrum",
                                 ha='center', va='center', transform=self.ax_spectrum.transAxes)


    def create_hr_tab(self):
        """Configure HR diagram tab"""
        self.hr_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.hr_tab, text="HR Diagram")

        self.fig_hr = Figure(figsize=(8, 6))
        self.ax_hr = self.fig_hr.add_subplot(111)
        self.canvas_hr = FigureCanvasTkAgg(self.fig_hr, master=self.hr_tab)
        self.canvas_hr.get_tk_widget().pack(expand=True, fill='both', padx=10, pady=10)
        self.ax_hr.text(0.5, 0.5, "HR Diagram will load with data",
                          ha='center', va='center', transform=self.ax_hr.transAxes)


    def create_research_tab(self):
        """Configure research tab with data table"""
        self.research_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.research_tab, text="Research")

        search_frame = ttk.Frame(self.research_tab)
        search_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(search_frame, text="Search:").pack(side='left', padx=5)
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side='left', fill='x', expand=True, padx=5)
        search_entry.bind("<KeyRelease>", self.filter_research_table)
        ttk.Button(search_frame, text="Clear", command=lambda: [self.search_var.set(""), self.filter_research_table()]).pack(side='left', padx=5)

        tree_frame = ttk.Frame(self.research_tab)
        tree_frame.pack(expand=True, fill='both', padx=10, pady=(0, 10))

        columns = ("Planet", "Star", "Period (days)", "Semi-Major (AU)", "Radius (Jup)", "Mass (Jup)", "Temp (K)", "Distance (pc)")
        self.research_tree = ttk.Treeview(tree_frame, columns=columns, show='headings')

        col_widths = {"Planet": 150, "Star": 150, "Period (days)": 100, "Semi-Major (AU)": 100, "Radius (Jup)": 100, "Mass (Jup)": 100, "Temp (K)": 100, "Distance (pc)": 100}
        for col in columns:
            self.research_tree.heading(col, text=col, command=lambda c=col: self.sort_research_table(c, False))
            self.research_tree.column(col, width=col_widths.get(col, 120), anchor='center')

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.research_tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.research_tree.xview)
        self.research_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.research_tree.pack(side="left", expand=True, fill='both')

        if not self.planetary_data.empty:
            self.populate_research_table(self.planetary_data)
        else:
             self.research_tree.insert("", "end", values=("No data loaded",)*len(columns))


    def populate_research_table(self, data_frame):
        """Fill research table with data from a DataFrame"""
        for item in self.research_tree.get_children():
            self.research_tree.delete(item)

        for _, row in data_frame.iterrows():
            values = (
                row.get('pl_name', "N/A"),
                row.get('hostname', "N/A"),
                f"{row.get('pl_orbper', np.nan):.3f}" if pd.notna(row.get('pl_orbper')) else "N/A",
                f"{row.get('pl_orbsmax', np.nan):.3f}" if pd.notna(row.get('pl_orbsmax')) else "N/A",
                f"{row.get('pl_radj', np.nan):.3f}" if pd.notna(row.get('pl_radj')) else "N/A",
                f"{row.get('pl_bmassj', np.nan):.3f}" if pd.notna(row.get('pl_bmassj')) else "N/A",
                f"{row.get('teff_gspphot', np.nan):.0f}" if pd.notna(row.get('teff_gspphot')) else "N/A", # Use new column name
                f"{row.get('sy_dist', np.nan):.1f}" if pd.notna(row.get('sy_dist')) else "N/A"
            )
            self.research_tree.insert("", "end", values=values)

    def filter_research_table(self, event=None):
        """Filter research table based on search query in Planet or Star column"""
        query = self.search_var.get().lower().strip()

        if not query:
            self.populate_research_table(self.planetary_data)
            return

        filtered_df = self.planetary_data[
            self.planetary_data['pl_name'].str.lower().str.contains(query, na=False) |
            self.planetary_data['hostname'].str.lower().str.contains(query, na=False)
        ]

        self.populate_research_table(filtered_df)

    def sort_research_table(self, col, reverse):
        """Sort research table by column"""
        data = [(self.research_tree.set(child, col), child) for child in self.research_tree.get_children('')]
        
        def sort_key(item):
            value = item[0]
            if value == "N/A":
                return -np.inf if reverse else np.inf
            try:
                return float(value)
            except ValueError:
                return value

        data.sort(key=sort_key, reverse=reverse)

        for index, (val, child) in enumerate(data):
            self.research_tree.move(child, '', index)

        self.research_tree.heading(col, command=lambda c=col: self.sort_research_table(c, not reverse))


    def update_planets(self, event=None):
        """Update planet list when star changes and update plots"""
        star = self.star_var.get()
        logging.info(f"Star selected: {star}")
        if self.planetary_data.empty or not star:
            self.planet_combo['values'] = []
            self.planet_var.set("")
            logging.warning("Planetary data is empty or no star selected, cannot update planets.")
            return

        planets = self.planetary_data[self.planetary_data['hostname'] == star]['pl_name'].tolist()
        self.planet_combo['values'] = planets
        if planets:
            self.planet_var.set(planets[0])
            logging.info(f"Updated planet list for {star}. Planets: {planets}")
        else:
             self.planet_var.set("")
             logging.warning(f"No planets found for star: {star}")

        self.plot_black_body_spectrum()
        self.plot_hr_diagram()
        if self.ani:
            self.ani.event_source.stop()
            self.ani = None
            self.ax_orbit.clear()
            self.ax_orbit.text(0.5, 0.5, "Select a planet and click 'Animate Orbit'", color='white',
                                 ha='center', va='center', transform=self.ax_orbit.transAxes)
            self.canvas_orbit.draw_idle()
            self.orbit_info.config(state='normal')
            self.orbit_info.delete(1.0, tk.END)
            self.orbit_info.config(state='disabled')


    def plot_black_body_spectrum(self):
        """Plot star's black body spectrum using local temperature data"""
        self.ax_spectrum.clear()
        star = self.star_var.get()

        if not star or self.planetary_data.empty:
            self.ax_spectrum.text(0.5, 0.5, "Select a star", ha='center', va='center', fontsize=12)
            self.canvas_spectrum.draw_idle()
            return

        star_data_rows = self.planetary_data[self.planetary_data['hostname'] == star]
        if star_data_rows.empty:
             logging.warning(f"No data found for star {star} in main dataframe.")
             self.ax_spectrum.text(0.5, 0.5, f"Data not found for {star}", ha='center', va='center', fontsize=12)
             self.canvas_spectrum.draw_idle()
             return

        star_data = star_data_rows.iloc[0]
        T = star_data.get('teff_gspphot') # Use new column name

        if pd.isna(T):
            self.ax_spectrum.text(0.5, 0.5, f"Stellar Temperature (st_teff)\ndata unavailable for {star}",
                                 ha='center', va='center', fontsize=12)
            self.canvas_spectrum.draw_idle()
            logging.warning(f"Temperature data unavailable for star: {star}")
            return

        try:
            T = float(T)
            wavelengths_nm = np.linspace(100, 3000, 500)
            wavelengths_m = wavelengths_nm * 1e-9

            intensity = (2 * H_PLANCK * C_LIGHT**2 / wavelengths_m**5) / \
                         (np.exp(H_PLANCK * C_LIGHT / (wavelengths_m * K_BOLTZMANN * T)) - 1)

            normalized_intensity = intensity / np.max(intensity) if np.max(intensity) > 0 else intensity

            self.ax_spectrum.plot(wavelengths_nm, normalized_intensity, color='orange', linewidth=2)
            self.ax_spectrum.set_title(f"Approx. Black Body Spectrum: {star} (T={T:.0f} K)", fontsize=12)
            self.ax_spectrum.set_xlabel("Wavelength (nm)", fontsize=10)
            self.ax_spectrum.set_ylabel("Normalized Intensity", fontsize=10)
            self.ax_spectrum.grid(True, linestyle=':', alpha=0.6)
            self.ax_spectrum.set_ylim(0, 1.1)
            self.ax_spectrum.set_xlim(min(wavelengths_nm), max(wavelengths_nm))
            logging.info(f"Plotted black body spectrum for {star} (T={T:.0f} K)")

        except Exception as e:
             logging.error(f"Error plotting black body spectrum for {star}: {e}", exc_info=True)
             self.ax_spectrum.text(0.5, 0.5, f"Error plotting spectrum\nfor {star}", ha='center', va='center', fontsize=12)

        self.canvas_spectrum.draw_idle()


    def plot_hr_diagram(self):
        """Plot HR diagram using local stellar data"""
        self.ax_hr.clear()

        if self.planetary_data.empty:
            self.ax_hr.text(0.5, 0.5, "Planetary data not loaded",
                           ha='center', va='center', fontsize=12)
            self.canvas_hr.draw_idle()
            return

        # Use new column names
        valid_data = self.planetary_data.dropna(subset=['teff_gspphot', 'lum_gspphot']).copy()
        valid_data['teff_gspphot'] = pd.to_numeric(valid_data['teff_gspphot'], errors='coerce')
        valid_data['lum_gspphot'] = pd.to_numeric(valid_data['lum_gspphot'], errors='coerce')
        valid_data = valid_data.dropna(subset=['teff_gspphot', 'lum_gspphot'])


        if valid_data.empty:
            self.ax_hr.text(0.5, 0.5, "Insufficient valid stellar data\n(Temp & Luminosity) for HR Diagram",
                           ha='center', va='center', fontsize=12)
            self.canvas_hr.draw_idle()
            logging.warning("Insufficient valid data for HR diagram.")
            return

        self.ax_hr.scatter(
            valid_data['teff_gspphot'],
            valid_data['lum_gspphot'], # Use direct luminosity
            alpha=0.4,
            s=20,
            label=f'All Host Stars ({len(valid_data)})',
            edgecolors='grey',
            linewidths=0.5
        )

        selected_star_name = self.star_var.get()
        if selected_star_name:
            star_data = valid_data[valid_data['hostname'] == selected_star_name]
            if not star_data.empty:
                star = star_data.iloc[0]
                self.ax_hr.scatter(
                    star['teff_gspphot'],
                    star['lum_gspphot'], # Use direct luminosity
                    color='red',
                    s=100,
                    edgecolor='black',
                    linewidth=1,
                    label=f'Selected: {selected_star_name}',
                    zorder=5
                )
                logging.info(f"Highlighted {selected_star_name} on HR diagram.")
            else:
                 logging.warning(f"Selected star {selected_star_name} not found in valid HR data.")


        self.ax_hr.set_yscale('log')
        self.ax_hr.set_xscale('log')
        self.ax_hr.invert_xaxis()
        self.ax_hr.set_title("Hertzsprung-Russell Diagram (Exoplanet Host Stars)", fontsize=12)
        self.ax_hr.set_xlabel("Effective Temperature (K)", fontsize=10)
        self.ax_hr.set_ylabel("Luminosity (Relative to Sun)", fontsize=10)
        if len(self.ax_hr.get_legend_handles_labels()[1]) > 0:
            self.ax_hr.legend(fontsize=9)
        self.ax_hr.grid(True, which="both", ls=":", alpha=0.5)
        logging.info("Plotted HR diagram.")
        self.canvas_hr.draw_idle()


    def animate_orbit(self):
        """Animate planet orbit using local data"""
        planet_name = self.planet_var.get()
        if not planet_name:
            messagebox.showwarning("No Planet Selected", "Please select a planet from the dropdown first.")
            return
        if self.planetary_data.empty:
            messagebox.showerror("Data Error", "Planetary data not loaded.")
            return

        if self.ani:
            try:
                self.ani.event_source.stop()
                logging.info("Stopped previous animation.")
            except AttributeError:
                pass
            self.ani = None 

        planet_data_rows = self.planetary_data[self.planetary_data['pl_name'] == planet_name]
        if planet_data_rows.empty:
            messagebox.showerror("Data Error", f"No data found for the selected planet: {planet_name}")
            logging.error(f"No data found for planet: {planet_name}")
            return

        p = planet_data_rows.iloc[0]

        try:
            a = pd.to_numeric(p.get('pl_orbsmax'), errors='coerce')
            period_days = pd.to_numeric(p.get('pl_orbper'), errors='coerce')
            ecc = pd.to_numeric(p.get('pl_orbeccen'), errors='coerce')

            if pd.isna(a) or pd.isna(period_days):
                messagebox.showerror("Orbit Data Error", f"Missing or invalid Semi-Major Axis (a={a}) or Orbital Period (P={period_days}) for {planet_name}.")
                logging.error(f"Missing/invalid orbit params for {planet_name}: a={a}, P={period_days}")
                return
            if pd.isna(ecc):
                ecc = 0.0
                logging.warning(f"Eccentricity missing for {planet_name}, assuming circular orbit (ecc=0.0).")

            a = float(a)
            period_days = float(period_days)
            ecc = float(ecc)
            b = a * np.sqrt(max(0, 1 - ecc**2))
            focus_offset = a * ecc

            theta_ellipse = np.linspace(0, 2 * np.pi, 200)
            x_ellipse_centered = a * np.cos(theta_ellipse)
            y_ellipse_centered = b * np.sin(theta_ellipse)
            x_orbit_path = x_ellipse_centered - focus_offset
            y_orbit_path = y_ellipse_centered

            self.orbit_info.config(state='normal')
            self.orbit_info.delete(1.0, tk.END)
            info_text = (
                f"Planet: {planet_name}\n"
                f"Host Star: {p.get('hostname', 'N/A')}\n"
                f"Orbital Period: {period_days:.3f} days\n"
                f"Semi-Major Axis: {a:.3f} AU\n"
                f"Eccentricity: {ecc:.3f}\n"
                f"Discovery Year: {int(p['disc_year']) if pd.notna(p.get('disc_year')) else 'N/A'}"
            )
            self.orbit_info.insert(tk.END, info_text)
            self.orbit_info.config(state='disabled')
            logging.info(f"Displaying orbit info for {planet_name}")

            self.ax_orbit.clear()
            self.ax_orbit.set_facecolor("black")
            self.ax_orbit.plot(x_orbit_path, y_orbit_path, 'w--', alpha=0.5, linewidth=1, label='Orbit Path')
            self.ax_orbit.plot(0, 0, 'o', color='gold', markersize=15, label=f'Star ({p.get("hostname", "")})')

            
            max_extent = a * (1 + ecc) + focus_offset
            plot_limit = max_extent * 1.2
            self.ax_orbit.set_xlim(-plot_limit, plot_limit)
            self.ax_orbit.set_ylim(-plot_limit, plot_limit)
            
            self.ax_orbit.set_aspect('equal', adjustable='box')
            self.ax_orbit.set_title(f"Orbit of {planet_name} (View from Above)", fontsize=12, color='white')
            self.ax_orbit.grid(True, linestyle=':', color='gray', alpha=0.4)
            if self.ax_orbit.get_legend_handles_labels()[1]:
                 legend = self.ax_orbit.legend(loc='upper right', fontsize=8)
                 plt.setp(legend.get_texts(), color='white')

            self.planet_dot, = self.ax_orbit.plot([], [], 'o', color='deepskyblue', markersize=7, label='Planet')
            self.dist_line, = self.ax_orbit.plot([], [], '--', color='red', alpha=0.6, linewidth=1)

            def init_anim():
                self.planet_dot.set_data([], [])
                self.dist_line.set_data([], [])
                return self.planet_dot, self.dist_line

            num_frames = 200
            def update_anim(frame):
                M = frame * (2 * np.pi / num_frames)
                try:
                    E = newton(lambda E_guess: E_guess - ecc * np.sin(E_guess) - M, M, tol=1e-6, maxiter=50)
                except RuntimeError:
                    E = M
                    if frame == 0: logging.warning(f"Newton solver failed for Kepler's Eq (ecc={ecc:.3f}), using E approx M.")

                x_planet_centered = a * np.cos(E)
                y_planet_centered = b * np.sin(E)
                x_planet = x_planet_centered - focus_offset
                y_planet = y_planet_centered

                self.planet_dot.set_data([x_planet], [y_planet])
                self.dist_line.set_data([0, x_planet], [0, y_planet])
                return self.planet_dot, self.dist_line

            self.ani = animation.FuncAnimation(
                self.fig_orbit,
                update_anim,
                init_func=init_anim,
                frames=num_frames,
                interval=max(20, int(period_days * 1000 / num_frames)),
                blit=False,
                repeat=True
            )

            self.canvas_orbit.draw_idle()
            logging.info(f"Started orbit animation for {planet_name}")

        except (ValueError, TypeError, KeyError) as e:
            messagebox.showerror("Animation Error", f"Could not animate orbit due to invalid data for {planet_name}:\n{str(e)}")
            logging.error(f"Invalid data for orbit animation {planet_name}: {e}", exc_info=True)
            self.ax_orbit.clear()
            self.ax_orbit.text(0.5, 0.5, f"Error animating orbit\nfor {planet_name}", color='white',
                                 ha='center', va='center', transform=self.ax_orbit.transAxes)
            self.canvas_orbit.draw_idle()
            self.orbit_info.config(state='normal')
            self.orbit_info.delete(1.0, tk.END)
            self.orbit_info.config(state='disabled')


    def show_atmosphere(self):
        """Display atmospheric composition popup (Transmission Spectrum + Molecular Abundance)"""
        planet_name = self.planet_var.get()
        if not planet_name:
            messagebox.showwarning("No Planet Selected", "Please select a planet first.")
            return

        popup = tk.Toplevel(self.root)
        popup.title(f"Atmospheric Composition - {planet_name}")
        popup.geometry("1000x800") 

        control_frame = ttk.Frame(popup, padding="10")
        control_frame.pack(fill='x', side='top')

        sig_frame = ttk.LabelFrame(control_frame, text="Show Atmospheric Signatures (Lines are approximate)")
        sig_frame.pack(fill='x', padx=5, pady=5)
 
        self.element_vars = {} 
        num_cols = 6 
        col_count = 0
        row_count = 0
        for element in ATMOSPHERIC_SIGNATURES.keys():
            var = tk.BooleanVar(value=False)
            chk = ttk.Checkbutton(sig_frame, text=element, variable=var, style="Toolbutton")
            chk.grid(row=row_count, column=col_count, padx=5, pady=2, sticky='w')
            self.element_vars[element] = var
            
            col_count += 1
            if col_count >= num_cols:
                col_count = 0
                row_count += 1
        
        plot_frame = ttk.Frame(popup)
        plot_frame.pack(expand=True, fill='both')

        notebook = ttk.Notebook(plot_frame)
        notebook.pack(expand=True, fill='both', padx=10, pady=(0, 10))

        spec_frame = ttk.Frame(notebook)
        mol_frame = ttk.Frame(notebook)
        notebook.add(spec_frame, text="Transmission Spectrum")
        notebook.add(mol_frame, text="Molecular Abundance")
        
        status_var = tk.StringVar(value="Loading data...")
        status_label = ttk.Label(control_frame, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(side='left', fill='x', expand=True, padx=(10, 0))

        def update_plots_in_popup():
            """Clear and redraw plot tabs based on checkbox states"""
            status_var.set(f"Fetching/Plotting for {planet_name}...")
            popup.update_idletasks() 
            for widget in spec_frame.winfo_children(): widget.destroy()
            for widget in mol_frame.winfo_children(): widget.destroy()
                 
            self.plot_transmission_spectrum(spec_frame, planet_name, self.element_vars) 
            self.plot_molecular_abundance(mol_frame, planet_name)
            
            status_var.set(f"Plots updated for {planet_name}.")

        update_button = ttk.Button(control_frame, text="Update Plot", command=update_plots_in_popup)
        update_button.pack(side='right', padx=10, pady=5)

        popup.after(100, update_plots_in_popup)


    def plot_transmission_spectrum(self, parent, planet_name, element_vars):
        """Plot transmission spectrum from MAST data in the specified parent widget."""
        logging.info(f"Plotting transmission spectrum for {planet_name}...")
        fig = Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)

        canvas_placeholder = FigureCanvasTkAgg(fig, master=parent)
        canvas_placeholder.get_tk_widget().pack(expand=True, fill='both', padx=10, pady=10)
        ax.text(0.5, 0.5, f"Fetching spectrum data for\n{planet_name} from MAST...\n(This can take a minute)",
               ha='center', va='center', fontsize=10)
        canvas_placeholder.draw()
        parent.update_idletasks()

        data = self.fetch_spectroscopy_data(planet_name)

        for widget in parent.winfo_children(): widget.destroy()
        ax.clear()

        if data.empty or 'wavelength' not in data.columns or 'flux' not in data.columns:
            logging.warning(f"No valid spectroscopy data to plot for {planet_name}.")
            ax.text(0.5, 0.5, f"No spectroscopy data available\nor failed to process FITS file for\n{planet_name}",
                   ha='center', va='center', fontsize=12)
        else:
            logging.info(f"Plotting {len(data)} data points for {planet_name}.")
            instrument = data['instrument'].iloc[0] if 'instrument' in data.columns else 'Unknown Instrument'
            has_errors = 'flux_err' in data.columns and data['flux_err'].notna().any()

            if has_errors:
                valid_error_data = data[pd.notna(data['flux_err']) & (data['flux_err'] > 0)].copy()
                if not valid_error_data.empty:
                    ax.errorbar(
                        valid_error_data['wavelength'],
                        valid_error_data['flux'],
                        yerr=valid_error_data['flux_err'],
                        fmt='o',
                        label=f"{instrument} (with errors)",
                        capsize=3,
                        alpha=0.7,
                        markersize=3,
                        ecolor='gray',
                        elinewidth=1
                    )
                    no_valid_error_data = data[~(pd.notna(data['flux_err']) & (data['flux_err'] > 0))].copy()
                    if not no_valid_error_data.empty:
                         ax.plot(
                             no_valid_error_data['wavelength'],
                             no_valid_error_data['flux'],
                             'o',
                             label=f"{instrument} (no valid errors)",
                             alpha=0.6,
                             markersize=3
                         )
                else:
                     has_errors = False

            if not has_errors:
                 ax.plot(
                     data['wavelength'],
                     data['flux'],
                     'o',
                     label=f"{instrument}",
                     alpha=0.7,
                     markersize=3
                 )
            
            target_unit = data['wavelength_unit'].iloc[0].lower() if 'wavelength_unit' in data.columns else 'um'
            plot_xmin, plot_xmax = ax.get_xlim()
            plotted_labels = set() 
            logging.info(f"Wavelength unit: {target_unit}, Plot X-limits: ({plot_xmin:.3f}, {plot_xmax:.3f})")

            if element_vars:
                for element, tk_var in element_vars.items():
                    if tk_var.get():
                        signature = ATMOSPHERIC_SIGNATURES[element] 
                        logging.debug(f"Checkbox '{element}' is checked. Plotting lines.")
                        
                        for wl_um in signature['wavelengths']:
                            converted_wl = wl_um
                            if 'nm' in target_unit or 'nanometer' in target_unit:
                                converted_wl *= 1000
                            elif 'angstrom' in target_unit:
                                converted_wl *= 10000
                            elif 'm' == target_unit:
                                converted_wl *= 1e-6

                            if plot_xmin <= converted_wl <= plot_xmax:
                                label = element if element not in plotted_labels else None
                                ax.axvline(
                                    x=converted_wl,
                                    color=signature['color'],
                                    linestyle=':',
                                    alpha=0.9,
                                    label=label,
                                    linewidth=1.5
                                )
                                plotted_labels.add(element)
                                logging.debug(f"Plotted '{element}' line at {converted_wl:.3f} {target_unit}")
                            else:
                                 logging.debug(f"Skipped '{element}' line at {wl_um} µm ({converted_wl:.3f} {target_unit}) - outside plot range.")
            else:
                 logging.warning("element_vars not passed correctly to plot_transmission_spectrum.")


            ax.set_title(f"Transmission Spectrum: {planet_name}", fontsize=12)
            ax.set_xlabel(f"Wavelength ({target_unit})", fontsize=10)
            ax.set_ylabel("Relative Flux / Transit Depth", fontsize=10) 
            
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                 ax.legend(handles, labels, fontsize=8, loc='best') 

            ax.grid(True, linestyle=':', alpha=0.5)
            logging.info(f"Finished plotting spectrum for {planet_name}.")

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill='both', padx=10, pady=10)
        
    def plot_molecular_abundance(self, parent, planet_name):
        """Plot molecular abundance bar chart from NASA Archive data."""
        logging.info(f"Plotting molecular abundance for {planet_name}...")
        fig = Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)

        # Fetch data (this uses the NEW local function)
        data = self.fetch_atmospheric_data(planet_name)

        if data.empty or 'molecule' not in data.columns or 'abundance' not in data.columns:
            logging.warning(f"No valid molecular abundance data to plot for {planet_name}.")
            ax.text(0.5, 0.5, f"No *detected* molecular abundance data found\n for {planet_name} in local CSV",
                   ha='center', va='center', fontsize=12)
        else:
             data['abundance'] = pd.to_numeric(data['abundance'], errors='coerce')
             plot_data = data.dropna(subset=['abundance', 'molecule']).copy()

             if plot_data.empty:
                  ax.text(0.5, 0.5, f"Molecular abundance data for\n{planet_name} is invalid.",
                          ha='center', va='center', fontsize=12)
             else:
                 molecules = plot_data['molecule'].tolist()
                 abundances = plot_data['abundance'].tolist()

                 bars = ax.bar(molecules, abundances, color='skyblue')
                 ax.set_title(f"Detected Molecular Abundance: {planet_name}", fontsize=12)
                 ax.set_ylabel("Detection (Placeholder)", fontsize=10) # Updated label
                 ax.tick_params(axis='x', rotation=45, labelsize=9)
                 ax.grid(True, axis='y', linestyle=':', alpha=0.6)
                 
                 # Set y-axis to just show the bars, since abundance is just '1'
                 ax.set_yticks([]) 
                 
                 logging.info(f"Plotted molecular abundance for {planet_name}.")

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill='both', padx=10, pady=10)

    def refresh_data(self):
        """This function is depreciated as we now load from a local file."""
        messagebox.showinfo("Local Data", f"The application is running on the local data file '{DATA_FILE}'.\n\nTo refresh the data, please download a new file from the NASA Exoplanet Archive and replace your existing CSV.")
        logging.info("Refresh button clicked, but app is in local mode. No action taken.")


# Main execution block
if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    available_themes = style.theme_names()
    logging.info(f"Available themes: {available_themes}")
    if 'clam' in available_themes:
         style.theme_use('clam')
         logging.info("Using 'clam' theme.")
    elif 'vista' in available_themes: # Good default on Windows
        style.theme_use('vista')
        logging.info("Using 'vista' theme.")

    app = ExoplanetAnalyzer(root)
    root.mainloop()
