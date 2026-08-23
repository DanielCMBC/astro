"""Runnable demonstration of the one-star-one-planet vertical slice.

Prints the full chain for HD 80606 b and, when ModernGL is available,
renders it offscreen to PNG files::

    python -m astro_explorer.app.slice_demo
    python -m astro_explorer.app.slice_demo --host WASP-39 --out frames/

Everything it shows comes from the committed local snapshot, so it runs with
the network disabled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import astropy.units as u
import numpy as np

from ..rendering.scene_builder import build_frame_scene
from .vertical_slice import PRIMARY_TARGET, build_slice, load_reference_catalog

__all__ = ["main", "report", "render_phases"]

#: Published time of periastron for HD 80606 b (Pearson et al. 2022).
HD80606B_PERIASTRON = 2458882.344


def report(system_slice, time_jd: float) -> str:
    """The scientific chain, as text."""
    lines = ["=" * 74, "VERTICAL SLICE: {0}".format(system_slice.frame.host_name), "=" * 74, ""]
    lines.extend(system_slice.describe_provenance())
    for record in system_slice.planets:
        lines.append("")
        lines.append("-" * 74)
        lines.extend(system_slice.describe_orbit(record, time_jd))
    return "\n".join(lines)


def kepler_second_law_table(system_slice, record, time_jd: float, intervals: int = 12) -> str:
    """Equal time steps and the distance covered in each.

    The point of the table is that the *areas* are equal while the arc
    lengths are not: on HD 80606 b the planet covers more than ten times as
    much ground near periapsis as it does near apoapsis in the same interval.
    """
    from ..physics.state_vectors import swept_area

    period = record.elements.period.value_in(u.day)
    if period is None:
        return "no period published; Kepler's second law cannot be demonstrated"

    substeps = 200
    times = time_jd + np.linspace(0.0, period, intervals * substeps + 1)
    positions = np.array([system_slice.state(record, t).position for t in times])

    areas = swept_area(positions).reshape(intervals, substeps).sum(axis=1)
    arcs = np.linalg.norm(np.diff(positions, axis=0), axis=1).reshape(intervals, substeps).sum(axis=1)
    radii = np.linalg.norm(positions[::substeps][:-1], axis=1)

    lines = [
        "",
        "KEPLER'S SECOND LAW - {0} equal time steps of {1:.3f} d".format(
            intervals, period / intervals
        ),
        "  {0:>4}  {1:>12}  {2:>14}  {3:>14}".format("step", "r (AU)", "arc (AU)", "area (AU^2)"),
    ]
    for index, (radius, arc, area) in enumerate(zip(radii, arcs, areas)):
        lines.append("  {0:>4}  {1:>12.6f}  {2:>14.6f}  {3:>14.8f}".format(index, radius, arc, area))

    lines.append(
        "  arc length varies by {0:.1f}x; swept area varies by {1:.2e} relative".format(
            arcs.max() / arcs.min(), float(np.std(areas) / np.mean(areas))
        )
    )
    return "\n".join(lines)


def render_phases(system_slice, record, time_jd: float, out_dir: Path, phases: int = 24):
    """Render the orbit with equal-time phase markers. Returns the file path."""
    from ..rendering.camera import Camera
    from ..rendering.gl_backend import GLRenderer, RenderSettings
    from ..rendering.renderer import RenderPlanet

    period = record.elements.period.value_in(u.day)
    scene = build_frame_scene(
        system_slice.frame,
        system_slice.star,
        system_slice.planets,
        mean_anomalies=system_slice.mean_anomalies(time_jd),
    )
    if not scene.orbits:
        return None

    points = scene.orbits[0].points_local.astype(np.float64)
    centre = 0.5 * (points.max(axis=0) + points.min(axis=0))
    extent = float(np.linalg.norm(points - centre, axis=1).max())

    for index in range(phases):
        state = system_slice.state(record, time_jd + (index / phases) * period)
        if state is None:
            break
        scene.planets.append(
            RenderPlanet(
                identifier="phase-{0}".format(index),
                position_local=system_slice.frame.place_planet(state.position).to_render(),
                radius_display=extent * 0.022,
                material_id="rocky",
                base_color=(1.0, 0.52, 0.18),
                emissive=1.4,
            )
        )

    settings = RenderSettings(width=1200, height=800, samples=8, orbit_line_width=2.4)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "{0}_equal_time_phases.png".format(record.name.replace(" ", "_"))

    with GLRenderer(settings) as renderer:
        camera = Camera(target=centre, aspect=settings.width / settings.height, pitch=1.25, yaw=0.25)
        camera.frame_object(extent, margin=1.3)
        renderer.save_png(renderer.render(scene, camera), path)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="HD 80606", help="host star in the snapshot")
    parser.add_argument("--planet", default=None, help="planet name (default: the innermost)")
    parser.add_argument("--time", type=float, default=None, help="epoch as a full JD")
    parser.add_argument("--out", default=None, help="directory for rendered PNGs")
    parser.add_argument("--no-render", action="store_true", help="text only")
    args = parser.parse_args(argv)

    system_slice = build_slice(args.host, load_reference_catalog())
    record = (
        system_slice.planet(args.planet)
        if args.planet
        else (system_slice.planet(PRIMARY_TARGET) or system_slice.planets[0])
    )
    if record is None:
        parser.error("planet not found in {0}".format(args.host))

    epoch = args.time
    if epoch is None:
        epoch = (
            record.elements.epoch_periastron.value_in(u.day)
            or record.elements.epoch_transit.value_in(u.day)
            or HD80606B_PERIASTRON
        )

    print(report(system_slice, epoch))
    print(kepler_second_law_table(system_slice, record, epoch))

    if args.no_render:
        return 0

    try:
        import moderngl  # noqa: F401
    except ImportError:
        print("\nModernGL is not installed; skipping the render.")
        print('Install it with:  pip install -e ".[render]"')
        return 0

    out_dir = Path(args.out) if args.out else Path("renders")
    try:
        path = render_phases(system_slice, record, epoch, out_dir)
    except Exception as exc:  # pragma: no cover - depends on the GL driver
        # Rendering was asked for and failed. Returning 0 here would let CI
        # report success having produced no image at all.
        print("\nRendering FAILED: {0}".format(exc), file=sys.stderr)
        return 1

    print("\nRendered -> {0}".format(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
