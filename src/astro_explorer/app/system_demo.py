"""Multi-planet ``SystemFrame`` rendering (review section 15).

Renders a whole host system from the committed offline snapshot::

    python -m astro_explorer.app.system_demo --host Kepler-11
    python -m astro_explorer.app.system_demo --host TRAPPIST-1 --frames 8
    python -m astro_explorer.app.system_demo --host HD 219134 --no-render

Everything the renderer receives is a prepared render state: a position, a
display radius and a material id. It is never handed orbital elements.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import astropy.units as u
import numpy as np

from ..rendering.scene_builder import build_frame_scene
from .vertical_slice import build_slice, load_reference_catalog

__all__ = ["main", "render_system", "frame_camera"]

#: Systems the committed snapshot carries, with a sensible viewing epoch.
DEFAULT_EPOCHS = {
    "Kepler-11": 2455590.0,
    "TRAPPIST-1": 2457000.0,
    "HD 219134": 2457000.0,
    "HD 80606": 2458882.344,
    "WASP-39": 2455343.0,
    "K2-18": 2457264.0,
}


def frame_camera(scene, aspect: float, *, pitch: float = 1.15, yaw: float = 0.35, margin: float = 1.35):
    """A camera that frames the whole system, centred on its orbits."""
    from ..rendering.camera import Camera

    points = [orbit.points_local.astype(np.float64) for orbit in scene.orbits]
    if points:
        stacked = np.vstack(points)
        centre = 0.5 * (stacked.max(axis=0) + stacked.min(axis=0))
        extent = float(np.linalg.norm(stacked - centre, axis=1).max())
    else:
        centre = np.zeros(3)
        extent = scene.bounding_radius()

    camera = Camera(target=centre, aspect=aspect, pitch=pitch, yaw=yaw)
    camera.frame_object(max(extent, 1e-6), margin=margin)
    return camera


def render_system(
    system_slice,
    time_jd: float,
    out_dir: Path,
    *,
    frames: int = 1,
    width: int = 1400,
    height: int = 900,
    labels: bool = True,
    span_periods: float = 1.0,
):
    """Render one system, optionally as a physical-time sequence.

    ``frames > 1`` steps forward over ``span_periods`` of the *outermost*
    planet's period, so the inner planets visibly lap the outer ones - the
    point of a physical clock rather than a normalised one.
    """
    from ..rendering.camera import Camera  # noqa: F401  (used via frame_camera)
    from ..rendering.gl_backend import GLRenderer, RenderSettings
    from ..rendering.labels import LabelStyle, draw_labels

    out_dir.mkdir(parents=True, exist_ok=True)
    settings = RenderSettings(width=width, height=height, samples=8, orbit_line_width=1.8)

    periods = [
        r.elements.period.value_in(u.day)
        for r in system_slice.planets
        if r.elements.period.is_known
    ]
    outer_period = max(periods) if periods else 1.0

    written = []
    with GLRenderer(settings) as renderer:
        # The camera is fixed across the sequence so the motion is the
        # planets', not the viewpoint's.
        first = build_frame_scene(
            system_slice.frame, system_slice.star, system_slice.planets,
            mean_anomalies=system_slice.mean_anomalies(time_jd),
        )
        camera = frame_camera(first, width / height)

        for index in range(max(1, frames)):
            epoch = time_jd + (index / max(1, frames)) * span_periods * outer_period
            scene = build_frame_scene(
                system_slice.frame, system_slice.star, system_slice.planets,
                mean_anomalies=system_slice.mean_anomalies(epoch),
            )
            scene.assign_lod(camera, height)
            image = renderer.render(scene, camera, time=index * 0.4)

            if labels:
                header = [
                    "{0}  -  JD {1:.3f}".format(system_slice.frame.host_name, epoch),
                    scene.annotations[0] if scene.annotations else "",
                ]
                image = draw_labels(
                    image,
                    scene.project_labels(camera, width, height),
                    LabelStyle(),
                    header=[line for line in header if line],
                )

            name = "{0}_{1:02d}.png".format(
                system_slice.frame.host_name.replace(" ", "_"), index
            )
            path = out_dir / name
            renderer.save_png(image, path)
            written.append(path)

        stats = {
            "planet_draw_calls": renderer.last_planet_draw_calls,
            "orbit_draw_calls": renderer.last_orbit_draw_calls,
            "lods": sorted({int(p.lod) for p in scene.planets}),
        }
    return written, stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="Kepler-11")
    parser.add_argument("--time", type=float, default=None, help="epoch as a full JD")
    parser.add_argument("--frames", type=int, default=1, help="physical-time sequence length")
    parser.add_argument("--periods", type=float, default=1.0, help="outer periods to span")
    parser.add_argument("--out", default="renders")
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args(argv)

    catalog = load_reference_catalog()
    system_slice = build_slice(args.host, catalog)
    epoch = args.time if args.time is not None else DEFAULT_EPOCHS.get(args.host, 2457000.0)

    print("=" * 78)
    print("\n".join(system_slice.describe_system(epoch)))
    print()
    print("\n".join(system_slice.describe_provenance()))

    if args.no_render:
        return 0

    try:
        import moderngl  # noqa: F401
    except ImportError:
        print('\nModernGL is not installed; skipping the render (pip install -e ".[render]").')
        return 0

    try:
        written, stats = render_system(
            system_slice, epoch, Path(args.out),
            frames=args.frames, labels=not args.no_labels,
            span_periods=args.periods,
        )
    except Exception as exc:  # pragma: no cover - depends on the GL driver
        # Rendering was asked for and failed. Returning 0 here would let CI
        # report success having produced no image at all.
        print("\nRendering FAILED: {0}".format(exc), file=sys.stderr)
        return 1

    print()
    print("Rendered {0} frame(s) -> {1}".format(len(written), written[0].parent))
    print(
        "  draw calls: {0} for {1} planets, {2} for {3} orbits; LOD levels in use {4}".format(
            stats["planet_draw_calls"], len(system_slice.planets),
            stats["orbit_draw_calls"], len(system_slice.planets), stats["lods"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
