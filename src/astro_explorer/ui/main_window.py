"""The corrected 2D Tkinter application (roadmap Phase 1).

Every scientific decision has moved out of this file and into the modules
under :mod:`astro_explorer`.  What remains here is widget wiring and layout.
That is the point of Phase 2: the same models now feed the 3D engine.

What changed relative to the original ``exoplanet_analyzer.py``:

* spectra are read by name from IPAC tables, and each measurement is plotted
  separately with its instrument and reference (3.1);
* the record-selection policy is explicit and shown on screen (3.2);
* a missing semimajor axis is derived or shown as unknown, never 1 AU (3.3);
* a missing eccentricity stays unknown; the circular assumption is labelled
  in the orbit view (3.4);
* the orbit animation runs on a physical clock with selectable time modes,
  not on frame numbers (3.5);
* the HR diagram plots luminosity, and the old temperature-radius plot keeps
  its own correctly named tab (3.6);
* the blackbody curve is labelled an ideal approximation (3.7);
* constants and units come from Astropy (3.8, 3.9);
* molecules are shown with detection status, instrument and reference rather
  than asserted as facts (3.12).
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402

from ..app.controller import Controller  # noqa: E402
from ..data.nasa_archive import SolutionPolicy  # noqa: E402
from ..physics.ephemeris import TimeMode  # noqa: E402
from .plots import (  # noqa: E402
    draw_blackbody,
    draw_hr_diagram,
    draw_orbit,
    draw_spectra,
    draw_temperature_radius_diagram,
)

__all__ = ["ExoplanetScientificSuite", "run"]

#: Animation frame interval in milliseconds.
FRAME_INTERVAL_MS = 40


class ExoplanetScientificSuite:
    """The Tkinter shell around the scientific core."""

    def __init__(self, root: tk.Tk, controller: Controller | None = None):
        self.root = root
        self.root.title("Exoplanet Scientific Suite - corrected 2D baseline")
        self.root.geometry("1500x950")

        self.controller = controller or Controller.create()
        self.state = self.controller.state
        self.animation = None
        self.signatures = []

        self._build_loading_ui()
        threading.Thread(target=self._load_data, daemon=True).start()

    # -- startup ---------------------------------------------------------
    def _build_loading_ui(self) -> None:
        self.loading_frame = ttk.Frame(self.root, padding=50)
        self.loading_frame.pack(expand=True)
        self.status_label = ttk.Label(
            self.loading_frame, text="Opening the local scientific snapshot...", font=("Arial", 13)
        )
        self.status_label.pack(pady=10)
        self.progress = ttk.Progressbar(
            self.loading_frame, orient="horizontal", length=440, mode="indeterminate"
        )
        self.progress.pack(pady=10)
        self.progress.start(12)

    def _set_status(self, text: str) -> None:
        self.root.after(0, lambda: self.status_label.config(text=text))

    def _load_data(self) -> None:
        """Offline first: local snapshot, then optionally synchronise."""
        if self.controller.load_local():
            self._set_status("Local snapshot loaded. Checking for updates...")
            self.root.after(0, self._build_main_ui)
            self._background_sync()
            return

        self._set_status("No local snapshot. Downloading from the NASA Exoplanet Archive...")
        result = self.controller.synchronize(SolutionPolicy.COMPOSITE)
        if result.succeeded:
            self.root.after(0, self._build_main_ui)
            return

        if self._load_legacy_cache():
            self.root.after(0, self._build_main_ui)
            return

        self._set_status("Could not load any catalogue: {0}".format(result.detail))
        self.root.after(0, self.progress.stop)

    def _background_sync(self) -> None:
        """Refresh in the background; a failure changes nothing on screen."""

        def worker() -> None:
            result = self.controller.synchronize()
            if result.succeeded:
                self.root.after(0, self._refresh_after_sync)

        threading.Thread(target=worker, daemon=True).start()

    def _load_legacy_cache(self) -> bool:
        """Fall back to the original ``exoplanet_cache.feather`` if present."""
        import pandas as pd

        for root in self.controller.resources.roots():
            candidate = Path(root) / "exoplanet_cache.feather"
            if not candidate.exists():
                continue
            try:
                frame = pd.read_feather(candidate)
            except Exception:
                continue
            self.controller.adopt_frame(
                frame,
                SolutionPolicy.COMPOSITE,
                "legacy cache {0} (provenance unknown)".format(candidate.name),
            )
            self.state.offline = True
            return True
        return False

    # -- main UI ---------------------------------------------------------
    def _build_main_ui(self) -> None:
        if getattr(self, "loading_frame", None) is not None:
            self.loading_frame.destroy()
            self.loading_frame = None

        self.signatures = self.controller.signatures()
        self._build_top_bar()
        self._build_notebook()

        hosts = self.state.hosts()
        if hosts:
            self.host_combo["values"] = hosts
            self.host_combo.current(0)
            self._on_host_selected()

    def _build_top_bar(self) -> None:
        bar = ttk.Frame(self.root, padding=8)
        bar.pack(side="top", fill="x")

        ttk.Label(bar, text="Host star:").grid(row=0, column=0, padx=(0, 4))
        self.host_var = tk.StringVar()
        self.host_combo = ttk.Combobox(
            bar, textvariable=self.host_var, width=24, state="readonly"
        )
        self.host_combo.grid(row=0, column=1, padx=4)
        self.host_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_host_selected())

        ttk.Label(bar, text="Planet:").grid(row=0, column=2, padx=(12, 4))
        self.planet_var = tk.StringVar()
        self.planet_combo = ttk.Combobox(
            bar, textvariable=self.planet_var, width=24, state="readonly"
        )
        self.planet_combo.grid(row=0, column=3, padx=4)

        ttk.Button(bar, text="Analyse", command=self.run_analysis).grid(row=0, column=4, padx=12)

        ttk.Label(bar, text="Time:").grid(row=0, column=5, padx=(16, 4))
        self.time_var = tk.StringVar(value=TimeMode.SCALED.label)
        time_combo = ttk.Combobox(
            bar,
            textvariable=self.time_var,
            width=28,
            state="readonly",
            values=[mode.label for mode in TimeMode],
        )
        time_combo.grid(row=0, column=6, padx=4)
        time_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_time_mode_changed())

        ttk.Label(bar, text="days/s:").grid(row=0, column=7, padx=(8, 2))
        self.scale_var = tk.StringVar(value="1")
        scale_entry = ttk.Entry(bar, textvariable=self.scale_var, width=8)
        scale_entry.grid(row=0, column=8)
        scale_entry.bind("<Return>", lambda _event: self._on_time_mode_changed())

        self.source_label = ttk.Label(bar, text=self.state.data_source_label)
        self.source_label.grid(row=1, column=0, columnspan=9, sticky="w", pady=(6, 0))

    def _build_notebook(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)

        self.tab_overview = ttk.Frame(self.notebook)
        self.tab_orbit = ttk.Frame(self.notebook)
        self.tab_hr = ttk.Frame(self.notebook)
        self.tab_radius = ttk.Frame(self.notebook)
        self.tab_blackbody = ttk.Frame(self.notebook)
        self.tab_spectra = ttk.Frame(self.notebook)
        self.tab_evidence = ttk.Frame(self.notebook)
        self.tab_data = ttk.Frame(self.notebook)

        for frame, title in (
            (self.tab_overview, " Overview "),
            (self.tab_orbit, " Orbit "),
            (self.tab_hr, " HR Diagram "),
            (self.tab_radius, " Temperature-Radius "),
            (self.tab_blackbody, " Black Body "),
            (self.tab_spectra, " Atmospheric Spectra "),
            (self.tab_evidence, " Molecular Evidence "),
            (self.tab_data, " Data & Provenance "),
        ):
            self.notebook.add(frame, text=title)

        self.overview_text = self._text_panel(self.tab_overview)
        self.evidence_text = self._text_panel(self.tab_evidence)
        self.data_text = self._text_panel(self.tab_data)

        self.fig_orbit, self.ax_orbit = self._figure(self.tab_orbit, "orbit")
        self.fig_hr, self.ax_hr = self._figure(self.tab_hr, "hr")
        self.fig_radius, self.ax_radius = self._figure(self.tab_radius, "radius")
        self.fig_blackbody, self.ax_blackbody = self._figure(self.tab_blackbody, "blackbody")
        self.fig_spectra, self.ax_spectra = self._figure(self.tab_spectra, "spectra")

        draw_hr_diagram(self.ax_hr, self.state.catalog)
        draw_temperature_radius_diagram(self.ax_radius, self.state.catalog)
        self.canvas_hr.draw()
        self.canvas_radius.draw()
        self._set_text(self.data_text, "\n".join(self.controller.data_report()))

    def _text_panel(self, parent: ttk.Frame) -> tk.Text:
        frame = ttk.Frame(parent, padding=12)
        frame.pack(fill="both", expand=True)
        widget = tk.Text(frame, wrap="none", font=("Consolas", 11), padx=10, pady=10)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=widget.yview)
        widget.configure(yscrollcommand=scroll.set, state="disabled")
        widget.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        return widget

    def _figure(self, parent: ttk.Frame, name: str):
        """Create a figure, embed it, and keep the canvas reachable.

        The canvas must stay referenced or Tk will garbage-collect the
        backing widget and the tab goes blank.
        """
        figure, axes = plt.subplots(figsize=(7, 6))
        figure.set_layout_engine("constrained")
        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        setattr(self, "canvas_{0}".format(name), canvas)
        return figure, axes

    @staticmethod
    def _set_text(widget: tk.Text, text: str) -> None:
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.config(state="disabled")

    # -- interaction -----------------------------------------------------
    def _on_host_selected(self) -> None:
        host = self.host_var.get()
        planets = self.state.planets_of(host)
        self.planet_combo["values"] = planets
        if planets:
            self.planet_combo.current(0)

    def _on_time_mode_changed(self) -> None:
        label = self.time_var.get()
        mode = next((m for m in TimeMode if m.label == label), TimeMode.SCALED)
        try:
            scale = float(self.scale_var.get())
        except ValueError:
            scale = 1.0
        self.state.set_time_mode(mode, scale_days_per_second=scale)
        if self.state.selected_planet:
            self.run_analysis()

    def _refresh_after_sync(self) -> None:
        self.source_label.config(text=self.state.data_source_label)
        self._set_text(self.data_text, "\n".join(self.controller.data_report()))

    def run_analysis(self) -> None:
        planet_name = self.planet_var.get()
        if not planet_name:
            return

        record = self.state.select(planet_name)
        if record is None:
            return

        self._update_overview(record)
        self._update_orbit(record)
        draw_hr_diagram(self.ax_hr, self.state.catalog, record)
        draw_temperature_radius_diagram(self.ax_radius, self.state.catalog, record)
        draw_blackbody(self.ax_blackbody, record)
        self.canvas_hr.draw()
        self.canvas_radius.draw()
        self.canvas_blackbody.draw()

        collection = self.controller.spectra_for(planet_name)
        draw_spectra(self.ax_spectra, collection, bands=self.signatures)
        self.canvas_spectra.draw()

        self._update_evidence(planet_name, collection)
        self._set_text(self.data_text, "\n".join(self.controller.data_report()))

    def _update_overview(self, record) -> None:
        lines = list(record.describe())

        traditional, vector = self.state.classification(record.name)
        lines += ["", "Classification:    {0}".format(traditional.describe())]
        lines += vector.describe()[1:]

        lines += [""] + self.controller.distance_lines(record.name)
        self._set_text(self.overview_text, "\n".join(lines))

    def _update_orbit(self, record) -> None:
        if self.animation is not None:
            self.animation.event_source.stop()
            self.animation = None

        elements = record.elements
        if not elements.semimajor_axis.is_known:
            draw_orbit(self.ax_orbit, record)
            self.canvas_orbit.draw()
            return

        controller = self.state.time

        def update(_frame):
            anomaly, _assumed = controller.mean_anomaly(elements, self.state.elapsed_seconds)
            draw_orbit(
                self.ax_orbit, record, anomaly, time_label=controller.describe()
            )
            return ()

        update(0)
        self.animation = animation.FuncAnimation(
            self.fig_orbit,
            update,
            interval=FRAME_INTERVAL_MS,
            blit=False,
            cache_frame_data=False,
            save_count=0,
        )
        self.canvas_orbit.draw()

    def _update_evidence(self, planet_name: str, collection) -> None:
        lines = ["Molecular evidence for {0}".format(planet_name), ""]
        lines += self.state.evidence.describe_planet(planet_name)
        lines += ["", "Local spectra ({0} measurement(s)):".format(len(collection))]
        if collection.is_empty:
            lines.append("  none found in the bundled tables directory")
        else:
            for spectrum in collection:
                lines.append("")
                lines.extend("  " + line for line in spectrum.describe())
        errors = getattr(collection, "errors", None)
        if errors:
            lines += ["", "Files that could not be parsed:"]
            lines += ["  " + error for error in errors]
        self._set_text(self.evidence_text, "\n".join(lines))


def run() -> None:
    """Entry point for ``astro-explorer-2d``."""
    root = tk.Tk()
    ExoplanetScientificSuite(root)
    root.mainloop()
