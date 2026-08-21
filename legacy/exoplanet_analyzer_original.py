# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import sys
import threading
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox, ttk


# --- SCIENTIFIC CONSTANTS ---
H_PLANCK = 6.62607015e-34
C_LIGHT = 299792458
K_BOLTZMANN = 1.380649e-23
B_WIEN = 2.897771955e-3
EARTH_TO_SOLAR = 3.003467e-6
JUPITER_TO_SOLAR = 9.547919e-4
SOLAR_EFFECTIVE_TEMPERATURE = 5772.0

NASA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
CATALOG_QUERY = """
select
    pl_name, hostname, pl_orbper, pl_orbsmax, pl_orbeccen,
    pl_rade, pl_radj, pl_bmasse, pl_bmassj, pl_eqt, pl_insol,
    st_teff, st_rad, st_mass, st_spectype,
    discoverymethod, disc_year, disc_facility, sy_dist
from pscomppars
order by hostname, pl_name
"""


def parse_float(value, default=np.nan):
    if pd.isna(value):
        return default
    text = str(value).replace("<", "").replace(">", "").strip()
    if not text or text.lower() == "null":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def solve_kepler(mean_anomaly, eccentricity):
    """Small Newton solver kept local so orbit animation does not depend on SciPy."""
    mean_anomaly = float(mean_anomaly)
    eccentricity = float(np.clip(eccentricity, 0.0, 0.98))
    eccentric_anomaly = mean_anomaly if eccentricity < 0.8 else np.pi

    for _ in range(12):
        residual = eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean_anomaly
        derivative = 1.0 - eccentricity * np.cos(eccentric_anomaly)
        if abs(derivative) < 1e-12:
            break
        eccentric_anomaly -= residual / derivative

    return eccentric_anomaly


def stellar_luminosity_solar(teff, radius):
    if not np.isfinite(teff) or not np.isfinite(radius) or teff <= 0 or radius <= 0:
        return np.nan
    return (radius**2) * (teff / SOLAR_EFFECTIVE_TEMPERATURE) ** 4


def classify_planet(row):
    radius_earth = parse_float(row.get("pl_rade"))
    radius_jupiter = parse_float(row.get("pl_radj"))
    mass_jupiter = parse_float(row.get("pl_bmassj"))
    equilibrium_temperature = parse_float(row.get("pl_eqt"))

    if np.isfinite(radius_earth):
        if radius_earth < 1.25:
            base = "Terrestrial"
        elif radius_earth < 2.0:
            base = "Super-Earth"
        elif radius_earth < 4.0:
            base = "Sub-Neptune"
        elif radius_earth < 8.0:
            base = "Neptune-like"
        else:
            base = "Gas giant"
    elif np.isfinite(radius_jupiter) and radius_jupiter >= 0.5:
        base = "Gas giant"
    elif np.isfinite(mass_jupiter) and mass_jupiter >= 0.3:
        base = "Gas giant"
    else:
        base = "Unknown"

    if np.isfinite(equilibrium_temperature):
        if equilibrium_temperature >= 1200:
            return f"Hot {base.lower()}"
        if equilibrium_temperature <= 180:
            return f"Cold {base.lower()}"
    return base


def format_value(value, unit="", digits=3):
    value = parse_float(value)
    if not np.isfinite(value):
        return "unknown"
    formatted = f"{value:.{digits}g}"
    return f"{formatted} {unit}".strip()


def format_year(value):
    value = parse_float(value)
    if not np.isfinite(value):
        return "unknown"
    return str(int(round(value)))


def format_text(value):
    if value is None or pd.isna(value) or not str(value).strip():
        return "unknown"
    return str(value)


def parse_spectrum_table(path):
    path = Path(path)
    metadata = {}
    columns = None
    rows = []

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("\\"):
                if "=" in line:
                    key, value = line[1:].split("=", 1)
                    metadata[key.strip()] = value.strip()
                continue

            if line.startswith("|"):
                if columns is None:
                    columns = [part.strip() for part in line.strip("|").split("|")]
                continue

            if columns is None:
                continue

            parts = line.split()
            values = {}
            for name in (
                "CENTRALWAVELNG",
                "BANDWIDTH",
                "PL_TRANDEP",
                "PL_TRANDEPERR1",
                "PL_TRANDEPERR2",
            ):
                if name not in columns:
                    values[name] = np.nan
                    continue
                index = columns.index(name)
                values[name] = parse_float(parts[index]) if index < len(parts) else np.nan

            wavelength = values["CENTRALWAVELNG"]
            transit_depth = values["PL_TRANDEP"]
            if not np.isfinite(wavelength) or not np.isfinite(transit_depth):
                continue

            rows.append(
                {
                    "wavelength_micron": wavelength,
                    "bandwidth_micron": values["BANDWIDTH"],
                    "transit_depth_percent": transit_depth,
                    "err_plus": values["PL_TRANDEPERR1"],
                    "err_minus": abs(values["PL_TRANDEPERR2"]),
                    "reference": metadata.get("REFERENCE", ""),
                    "instrument": metadata.get("INSTRUMENT", ""),
                    "source_file": path.name,
                }
            )

    return pd.DataFrame(rows)


class ExoplanetScientificSuite:
    def __init__(self, root):
        self.root = root
        self.root.title("NASA Exoplanet Scientific Suite")
        self.root.geometry("1400x900")

        self.ani = None
        self.df = pd.DataFrame()
        self.data_source = "NASA Exoplanet Archive"
        self.source_dir = Path(__file__).resolve().parent
        self.bundle_dir = Path(getattr(sys, "_MEIPASS", self.source_dir))
        self.cache_path = self.source_dir / "exoplanet_cache.feather"

        self.load_local_assets()
        self.setup_loading_ui()
        threading.Thread(target=self.fetch_and_organize_data, daemon=True).start()

    def data_candidates(self, filename):
        seen = set()
        for base in (self.source_dir, self.bundle_dir, Path.cwd()):
            for candidate in (base / filename, base / "tables" / filename):
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield resolved

    def resolve_data_file(self, filename):
        for candidate in self.data_candidates(filename):
            if candidate.exists():
                return candidate
        return None

    def spectra_directories(self):
        seen = set()
        for base in (self.source_dir, self.bundle_dir, Path.cwd()):
            candidate = (base / "tables").resolve()
            if candidate.exists() and candidate not in seen:
                seen.add(candidate)
                yield candidate

    def load_local_assets(self):
        mols_path = self.resolve_data_file("planet_molecules.csv")
        if mols_path:
            self.mols = pd.read_csv(mols_path, comment="#", encoding="utf-8")
            return

        self.mols = pd.DataFrame(
            [
                {"pl_name": "WASP-39 b", "molecule": "CO2 (Carbon Dioxide)", "ref_url": ""},
                {"pl_name": "WASP-39 b", "molecule": "H2O (Water)", "ref_url": ""},
                {"pl_name": "WASP-96 b", "molecule": "H2O (Water)", "ref_url": ""},
                {"pl_name": "K2-18 b", "molecule": "CH4 (Methane)", "ref_url": ""},
                {"pl_name": "K2-18 b", "molecule": "CO2 (Carbon Dioxide)", "ref_url": ""},
            ]
        )

    def setup_loading_ui(self):
        self.loading_frame = ttk.Frame(self.root, padding=50)
        self.loading_frame.pack(expand=True)
        self.status_label = ttk.Label(
            self.loading_frame,
            text="Downloading one-row-per-planet catalog from NASA TAP...",
            font=("Arial", 14),
        )
        self.status_label.pack(pady=10)
        self.progress = ttk.Progressbar(
            self.loading_frame, orient="horizontal", length=400, mode="indeterminate"
        )
        self.progress.pack(pady=10)
        self.progress.start(10)

    def fetch_catalog_from_nasa(self):
        response = requests.get(
            NASA_TAP_URL,
            params={"query": " ".join(CATALOG_QUERY.split()), "format": "csv"},
            timeout=45,
        )
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text))

    def fetch_and_organize_data(self):
        try:
            self.root.after(0, lambda: self.status_label.config(text="Fetching NASA composite parameters..."))
            df = self.fetch_catalog_from_nasa()
            self.data_source = "NASA Exoplanet Archive pscomppars"
            self.df = self.prepare_catalog(df)
            self.save_cache()
            self.root.after(0, self.build_main_ui)
        except Exception as exc:
            cached = self.load_cache()
            if cached is not None:
                self.df = cached
                self.data_source = f"local cache ({self.cache_path.name})"
                self.root.after(0, self.build_main_ui)
                return

            self.root.after(0, lambda: self.status_label.config(text=f"Connection Error: {exc}"))
            self.root.after(0, self.progress.stop)

    def prepare_catalog(self, df):
        required_columns = [
            "pl_name",
            "hostname",
            "pl_orbper",
            "pl_orbsmax",
            "pl_orbeccen",
            "pl_rade",
            "pl_radj",
            "pl_bmasse",
            "pl_bmassj",
            "pl_eqt",
            "pl_insol",
            "st_teff",
            "st_rad",
            "st_mass",
            "st_spectype",
            "discoverymethod",
            "disc_year",
            "disc_facility",
            "sy_dist",
        ]
        for column in required_columns:
            if column not in df.columns:
                df[column] = np.nan

        numeric_columns = [
            "pl_orbper",
            "pl_orbsmax",
            "pl_orbeccen",
            "pl_rade",
            "pl_radj",
            "pl_bmasse",
            "pl_bmassj",
            "pl_eqt",
            "pl_insol",
            "st_teff",
            "st_rad",
            "st_mass",
            "disc_year",
            "sy_dist",
        ]
        for column in numeric_columns:
            if column in df.columns:
                df[column] = df[column].map(parse_float)

        df["pl_orbeccen"] = df["pl_orbeccen"].fillna(0.0).clip(lower=0.0, upper=0.98)
        df["pl_orbsmax"] = df["pl_orbsmax"].where(df["pl_orbsmax"] > 0)
        df["pl_mass_solar"] = df["pl_bmassj"] * JUPITER_TO_SOLAR
        mass_mask = df["pl_mass_solar"].isna()
        df.loc[mass_mask, "pl_mass_solar"] = df.loc[mass_mask, "pl_bmasse"] * EARTH_TO_SOLAR

        df["st_luminosity_solar"] = [
            stellar_luminosity_solar(teff, radius)
            for teff, radius in zip(df["st_teff"], df["st_rad"])
        ]
        derived_flux = df["st_luminosity_solar"] / (df["pl_orbsmax"] ** 2)
        df["pl_flux_earth"] = df["pl_insol"].where(df["pl_insol"].notna(), derived_flux)
        df["planet_class"] = df.apply(classify_planet, axis=1)

        return df.sort_values(["hostname", "pl_name"]).reset_index(drop=True)

    def save_cache(self):
        try:
            self.df.to_feather(self.cache_path)
        except Exception:
            pass

    def load_cache(self):
        if not self.cache_path.exists():
            return None
        try:
            return self.prepare_catalog(pd.read_feather(self.cache_path))
        except Exception:
            return None

    def build_main_ui(self):
        self.loading_frame.destroy()

        top_bar = ttk.Frame(self.root, padding=10)
        top_bar.pack(side="top", fill="x")

        ttk.Label(top_bar, text="Host Star:").grid(row=0, column=0, padx=5)
        self.star_var = tk.StringVar()
        self.star_combo = ttk.Combobox(top_bar, textvariable=self.star_var, width=25, state="readonly")
        self.star_combo["values"] = sorted(self.df["hostname"].dropna().unique().tolist())
        self.star_combo.grid(row=0, column=1, padx=5)
        self.star_combo.bind("<<ComboboxSelected>>", self.update_planet_list)

        ttk.Label(top_bar, text="Planet:").grid(row=0, column=2, padx=5)
        self.planet_var = tk.StringVar()
        self.planet_combo = ttk.Combobox(top_bar, textvariable=self.planet_var, width=25, state="readonly")
        self.planet_combo.grid(row=0, column=3, padx=5)

        ttk.Button(top_bar, text="Analyse", command=self.run_analysis).grid(row=0, column=4, padx=15)
        ttk.Label(top_bar, text=f"Data: {self.data_source}").grid(row=0, column=5, padx=5, sticky="w")

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_summary = ttk.Frame(self.nb)
        self.tab_orbit = ttk.Frame(self.nb)
        self.tab_hr = ttk.Frame(self.nb)
        self.tab_bb = ttk.Frame(self.nb)
        self.tab_atmo = ttk.Frame(self.nb)

        self.nb.add(self.tab_summary, text=" Overview ")
        self.nb.add(self.tab_orbit, text=" Orbit ")
        self.nb.add(self.tab_hr, text=" HR Diagram ")
        self.nb.add(self.tab_bb, text=" Black Body ")
        self.nb.add(self.tab_atmo, text=" Atmospheric Spectra ")

        self.setup_summary_tab()
        self.setup_orbit_tab()
        self.setup_hr_tab()
        self.setup_bb_tab()
        self.setup_atmo_tab()

        if self.star_combo["values"]:
            self.star_combo.current(0)
            self.update_planet_list()

    def setup_summary_tab(self):
        frame = ttk.Frame(self.tab_summary, padding=18)
        frame.pack(fill="both", expand=True)
        self.summary_text = tk.Text(frame, wrap="word", height=18, font=("Consolas", 12), padx=12, pady=12)
        self.summary_text.pack(fill="both", expand=True)
        self.set_summary_text("Choose a system and run analysis to inspect the selected planet.")

    def set_summary_text(self, text):
        self.summary_text.config(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", text)
        self.summary_text.config(state="disabled")

    def setup_orbit_tab(self):
        self.fig_orb, self.ax_orb = plt.subplots(figsize=(6, 6))
        self.canvas_orb = FigureCanvasTkAgg(self.fig_orb, master=self.tab_orbit)
        self.canvas_orb.get_tk_widget().pack(fill="both", expand=True)

    def setup_hr_tab(self):
        self.fig_hr, self.ax_hr = plt.subplots(figsize=(6, 6))
        self.canvas_hr = FigureCanvasTkAgg(self.fig_hr, master=self.tab_hr)
        self.canvas_hr.get_tk_widget().pack(fill="both", expand=True)

        valid_stars = self.df.dropna(subset=["st_teff", "st_luminosity_solar"]).drop_duplicates(
            subset=["hostname"]
        )
        self.ax_hr.scatter(
            valid_stars["st_teff"],
            valid_stars["st_luminosity_solar"],
            s=4,
            color="gray",
            alpha=0.3,
            label="Known hosts",
        )
        self.ax_hr.set_yscale("log")
        self.ax_hr.invert_xaxis()
        self.ax_hr.set_title("Stellar Luminosity vs Effective Temperature")
        self.ax_hr.set_xlabel("Effective Temperature (K)")
        self.ax_hr.set_ylabel("Luminosity (L_sun)")
        self.ax_hr.grid(True, linestyle=":", alpha=0.4)

        (self.hr_highlight,) = self.ax_hr.plot(
            [], [], "ro", markersize=10, markeredgecolor="black", label="Selected star"
        )
        self.ax_hr.legend()

    def setup_bb_tab(self):
        self.fig_bb, self.ax_bb = plt.subplots(figsize=(6, 6))
        self.canvas_bb = FigureCanvasTkAgg(self.fig_bb, master=self.tab_bb)
        self.canvas_bb.get_tk_widget().pack(fill="both", expand=True)

    def setup_atmo_tab(self):
        self.fig_atmo, self.ax_atmo = plt.subplots(figsize=(6, 6))
        self.canvas_atmo = FigureCanvasTkAgg(self.fig_atmo, master=self.tab_atmo)
        self.canvas_atmo.get_tk_widget().pack(fill="both", expand=True)
        self.ax_atmo.text(0.5, 0.5, "No atmospheric spectrum loaded", ha="center", va="center", fontsize=12)

    def update_planet_list(self, event=None):
        star = self.star_var.get()
        planets = sorted(self.df[self.df["hostname"] == star]["pl_name"].tolist())
        self.planet_combo["values"] = planets
        if planets:
            self.planet_combo.current(0)

    def run_analysis(self):
        p_name = self.planet_var.get()
        if not p_name:
            return

        p_data = self.df[self.df["pl_name"] == p_name].iloc[0]

        self.update_summary(p_data)
        self.draw_orbit(p_data)
        self.update_hr_diagram(p_data)
        self.draw_black_body(p_data)
        self.auto_plot_spectra(p_name)

        atmo_data = self.mols[self.mols["pl_name"] == p_name]
        if not atmo_data.empty:
            mols_found = ", ".join(atmo_data["molecule"].dropna().tolist())
            messagebox.showinfo("Atmosphere Data", f"Atmospheric detections:\n\n{mols_found}")

    def update_summary(self, p_data):
        lines = [
            f"Planet:              {p_data['pl_name']}",
            f"Host star:           {p_data['hostname']}",
            f"Planet class:        {p_data['planet_class']}",
            f"Discovery:           {format_text(p_data.get('discoverymethod'))} ({format_year(p_data.get('disc_year'))})",
            f"Facility:            {format_text(p_data.get('disc_facility'))}",
            "",
            f"Orbital period:      {format_value(p_data.get('pl_orbper'), 'days')}",
            f"Semi-major axis:     {format_value(p_data.get('pl_orbsmax'), 'AU')}",
            f"Eccentricity:        {format_value(p_data.get('pl_orbeccen'), digits=3)}",
            f"Equilibrium temp.:   {format_value(p_data.get('pl_eqt'), 'K')}",
            f"Insolation:          {format_value(p_data.get('pl_flux_earth'), 'Earth flux')}",
            "",
            f"Radius:              {format_value(p_data.get('pl_rade'), 'R_earth')} / {format_value(p_data.get('pl_radj'), 'R_jup')}",
            f"Mass:                {format_value(p_data.get('pl_bmasse'), 'M_earth')} / {format_value(p_data.get('pl_bmassj'), 'M_jup')}",
            "",
            f"Host temperature:    {format_value(p_data.get('st_teff'), 'K')}",
            f"Host radius:         {format_value(p_data.get('st_rad'), 'R_sun')}",
            f"Host mass:           {format_value(p_data.get('st_mass'), 'M_sun')}",
            f"Host luminosity:     {format_value(p_data.get('st_luminosity_solar'), 'L_sun')}",
            f"Spectral type:       {format_text(p_data.get('st_spectype'))}",
            f"System distance:     {format_value(p_data.get('sy_dist'), 'pc')}",
        ]
        self.set_summary_text("\n".join(lines))

    def spectrum_files_for_planet(self, p_name):
        file_prefix = p_name.replace(" ", "_").replace("-", "_")
        found_files = []
        for directory in self.spectra_directories():
            found_files.extend(directory.glob(f"{file_prefix}*.tbl"))
        return sorted(set(path.resolve() for path in found_files))

    def auto_plot_spectra(self, p_name):
        self.ax_atmo.clear()

        found_files = self.spectrum_files_for_planet(p_name)
        if not found_files:
            self.ax_atmo.text(
                0.5,
                0.5,
                f"No local spectra data found for {p_name}",
                color="red",
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=12,
            )
            self.ax_atmo.set_title(f"Atmospheric Spectra for {p_name}")
            self.canvas_atmo.draw()
            return

        plotted = 0
        for file_path in found_files:
            try:
                spectrum = parse_spectrum_table(file_path)
            except Exception as exc:
                print(f"Error reading {file_path}: {exc}")
                continue

            if spectrum.empty:
                continue

            x_values = spectrum["wavelength_micron"].to_numpy()
            y_values = spectrum["transit_depth_percent"].to_numpy()
            err_plus = spectrum["err_plus"].abs().to_numpy()
            err_minus = spectrum["err_minus"].abs().to_numpy()
            has_errors = np.isfinite(err_plus).any() and np.isfinite(err_minus).any()

            label = spectrum["reference"].iloc[0] or file_path.name
            if has_errors:
                yerr = np.vstack(
                    [
                        np.where(np.isfinite(err_minus), err_minus, 0.0),
                        np.where(np.isfinite(err_plus), err_plus, 0.0),
                    ]
                )
                self.ax_atmo.errorbar(
                    x_values,
                    y_values,
                    yerr=yerr,
                    fmt="o",
                    ms=3,
                    alpha=0.7,
                    capsize=2,
                    label=label,
                )
            else:
                self.ax_atmo.plot(x_values, y_values, "o", ms=3, alpha=0.7, label=label)
            plotted += 1

        if plotted:
            self.ax_atmo.set_title(f"Observed Transmission Spectra: {p_name}")
            self.ax_atmo.set_xlabel("Wavelength (microns)")
            self.ax_atmo.set_ylabel("Transit Depth (%)")
            self.ax_atmo.grid(True, linestyle=":", alpha=0.6)
            if plotted <= 5:
                self.ax_atmo.legend()
        else:
            self.ax_atmo.text(
                0.5,
                0.5,
                "Spectra files found, but no usable transmission data was parsed",
                color="red",
                ha="center",
                va="center",
                fontweight="bold",
            )

        self.canvas_atmo.draw()

    def draw_orbit(self, p_data):
        a = parse_float(p_data.get("pl_orbsmax"), default=1.0)
        if not np.isfinite(a) or a <= 0:
            a = 1.0
        e = parse_float(p_data.get("pl_orbeccen"), default=0.0)
        e = float(np.clip(e, 0.0, 0.98))

        self.ax_orb.clear()
        self.ax_orb.set_facecolor("#000000")
        self.ax_orb.plot(0, 0, "yo", markersize=12, label="Host Star")

        nu_path = np.linspace(0, 2 * np.pi, 500)
        r_path = (a * (1 - e**2)) / (1 + e * np.cos(nu_path))
        self.ax_orb.plot(r_path * np.cos(nu_path), r_path * np.sin(nu_path), "w--", alpha=0.3)

        (planet_dot,) = self.ax_orb.plot([], [], "ro", markersize=8, label=p_data["pl_name"])

        def update(frame):
            mean_anomaly = (2 * np.pi * frame) / 160
            eccentric_anomaly = solve_kepler(mean_anomaly, e)
            true_anomaly = 2 * np.arctan2(
                np.sqrt(1 + e) * np.sin(eccentric_anomaly / 2),
                np.sqrt(1 - e) * np.cos(eccentric_anomaly / 2),
            )
            radius = (a * (1 - e**2)) / (1 + e * np.cos(true_anomaly))
            x_pos, y_pos = radius * np.cos(true_anomaly), radius * np.sin(true_anomaly)
            planet_dot.set_data([x_pos], [y_pos])
            self.ax_orb.set_title(f"Distance: {radius:.4f} AU", color="white")
            return (planet_dot,)

        limit = max(a * (1 + e) * 1.25, 0.02)
        self.ax_orb.set_xlim(-limit, limit)
        self.ax_orb.set_ylim(-limit, limit)
        self.ax_orb.set_aspect("equal")
        self.ax_orb.tick_params(colors="white")

        if self.ani:
            self.ani.event_source.stop()
        self.ani = animation.FuncAnimation(self.fig_orb, update, frames=160, interval=40, blit=False)
        self.canvas_orb.draw()

    def update_hr_diagram(self, p_data):
        teff = parse_float(p_data.get("st_teff"))
        luminosity = parse_float(p_data.get("st_luminosity_solar"))

        if np.isfinite(teff) and np.isfinite(luminosity):
            self.hr_highlight.set_data([teff], [luminosity])
            self.hr_highlight.set_label(f"{p_data['hostname']} (T={teff:.0f} K, L={luminosity:.3g} L_sun)")
            self.ax_hr.legend()
        else:
            self.hr_highlight.set_data([], [])
        self.canvas_hr.draw()

    def draw_black_body(self, p_data):
        temperature = parse_float(p_data.get("st_teff"))
        self.ax_bb.clear()

        if np.isfinite(temperature) and temperature > 0:
            wavelength = np.linspace(100e-9, 3000e-9, 1000)
            exponent = (H_PLANCK * C_LIGHT) / (wavelength * K_BOLTZMANN * temperature)
            intensity = (2 * H_PLANCK * C_LIGHT**2) / (wavelength**5 * np.expm1(exponent))
            intensity = intensity / np.nanmax(intensity)

            self.ax_bb.plot(wavelength * 1e9, intensity, color="orange", lw=2)
            self.ax_bb.fill_between(wavelength * 1e9, intensity, color="orange", alpha=0.3)
            self.ax_bb.set_title(f"Black Body Spectrum for {p_data['hostname']} (T = {temperature:.0f} K)")
            self.ax_bb.set_xlabel("Wavelength (nm)")
            self.ax_bb.set_ylabel("Relative Spectral Radiance")
            self.ax_bb.grid(True, linestyle=":", alpha=0.6)

            peak_wavelength = (B_WIEN / temperature) * 1e9
            self.ax_bb.axvline(
                peak_wavelength,
                color="red",
                linestyle="--",
                label=f"Peak Wavelength: {peak_wavelength:.1f} nm",
            )
            self.ax_bb.legend()
        else:
            self.ax_bb.text(0.5, 0.5, "Temperature data not available.", ha="center", va="center", color="red")

        self.canvas_bb.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = ExoplanetScientificSuite(root)
    root.mainloop()
