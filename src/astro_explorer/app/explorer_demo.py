"""Fly from the stellar neighbourhood into a host system.

Review section 10. Renders the approach as a sequence, so the frame
transition, the picking and the labels can be inspected as pixels rather
than only as assertions::

    python -m astro_explorer.app.explorer_demo
    python -m astro_explorer.app.explorer_demo --host Kepler-11 --frames 6
    python -m astro_explorer.app.explorer_demo --no-render

Each frame is annotated with the active reference frame and the distance to
the host, so the switch from parsecs to AU is visible in the output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import astropy.units as u
import numpy as np

from .explorer import (
    Explorer,
    UniverseTarget,
    UnknownSystemPositionError,
    ViewState,
)
from .time_controls import TimeControls
from .vertical_slice import build_slice, load_reference_catalog

__all__ = ["main", "build_explorer", "render_approach"]

DEFAULT_EPOCHS = {
    "HD 80606": 2458882.344,
    "Kepler-11": 2455590.0,
    "TRAPPIST-1": 2457000.0,
    "HD 219134": 2457000.0,
    "WASP-39": 2455343.0,
    "K2-18": 2457264.0,
}


def build_explorer(catalog) -> Explorer:
    """An explorer holding every host the snapshot can actually place.

    A host without a usable distance has no position in the neighbourhood
    view, so it is left out rather than placed somewhere convenient.
    """
    targets: list[UniverseTarget] = []
    skipped: list[str] = []

    for host in sorted(catalog["hostname"].unique()):
        system = build_slice(host, catalog)
        star = system.star
        if star.position is None or not star.position.has_distance:
            skipped.append(host)
            continue
        targets.append(
            UniverseTarget(
                name=host,
                position_pc=star.position.cartesian_pc(),
                temperature_k=star.effective_temperature.value_in(u.K),
                radius_solar=star.radius.value_in(u.R_sun),
                distance_pc=star.position.distance.value_in(u.pc),
                planet_count=len(system.planets),
            )
        )

    explorer = Explorer(targets=targets)
    explorer.move_to_pc([0.0, 0.0, 40.0])
    setattr(explorer, "skipped_hosts", skipped)
    return explorer


def render_approach(
    explorer: Explorer,
    epoch: float,
    out_dir: Path,
    *,
    frames: int = 5,
    width: int = 1400,
    height: int = 900,
) -> list[Path]:
    """Render the approach, one frame per waypoint."""
    from ..rendering.gl_backend import GLRenderer, RenderSettings
    from ..rendering.labels import LabelStyle, draw_labels

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    waypoints = explorer.path_to_system(frames)

    with GLRenderer(RenderSettings(width=width, height=height, samples=4)) as renderer:
        for index, position in enumerate(waypoints):
            explorer.move_to_pc(position)
            scene = explorer.scene(epoch)

            frame = explorer.active_frame
            local = explorer.camera_position.values
            radius = float(np.linalg.norm(local))

            # The explorer has already aimed the camera at the focused
            # system; only the framing is the demo's business.
            camera = explorer.camera
            camera.aspect = width / height
            camera.pitch = 0.55
            camera.yaw = 0.35
            scene.assign_lod(camera, height)

            image = renderer.render(scene, camera, time=index * 0.3)
            header = [
                "{0}  frame {1}/{2}".format(
                    explorer.view.label, index + 1, len(waypoints)
                ),
                "active frame: {0}    camera {1:.4g} {2} from origin".format(
                    frame.describe(), radius, frame.unit_label
                ),
                "distance to {0}: {1:.5f} pc    engages within {2:.4f} pc".format(
                    explorer.system.host_name,
                    explorer.distance_to_system_pc(),
                    explorer.system.engage_radius_pc,
                ),
            ]
            image = draw_labels(
                image, explorer.labels(scene, width, height), LabelStyle(), header=header
            )

            path = out_dir / "approach_{0:02d}_{1}.png".format(index, explorer.view.value.lower())
            renderer.save_png(image, path)
            written.append(path)

    return written


def _print_panels(explorer: Explorer, system, epoch: float) -> None:
    """Show the system summary and one selected planet, with provenance."""
    print()
    print("SYSTEM PANEL")
    panel = explorer.panel(epoch)
    if panel is not None:
        print("\n".join("  " + line for line in panel.render()))

    if not system.planets:
        return

    record = system.planets[0]
    if record.entity_id is None:
        return

    explorer.select(str(record.entity_id), "planet")
    print()
    print("SELECTED PLANET PANEL  (selection generation {0})".format(explorer.generation))
    planet_panel = explorer.panel(epoch)
    if planet_panel is not None:
        print("\n".join("  " + line for line in planet_panel.render()))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="HD 80606")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--time", type=float, default=None)
    parser.add_argument("--out", default="renders")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args(argv)

    catalog = load_reference_catalog()
    explorer = build_explorer(catalog)
    system = build_slice(args.host, catalog)

    print("=" * 78)
    print("EXPLORER: neighbourhood -> {0}".format(args.host))
    print("=" * 78)

    # A host with no published distance has no place in the neighbourhood
    # view and cannot be flown to. It opens detached instead: the star is
    # the origin of its own frame, and no galactic position is claimed.
    detached = False
    try:
        explorer.focus(args.host, system.planets, system.star)
    except UnknownSystemPositionError as exc:
        print("Cannot fly to {0}: {1}".format(args.host, exc))
        print("Opening it detached instead.")
        print()
        explorer.open_detached(args.host, system.planets, system.star)
        detached = True

    controls = TimeControls.for_system(system.planets)
    if args.time is not None:
        controls.seek(args.time, rebase=True)
    elif args.host in DEFAULT_EPOCHS and any(r.elements.dated_epochs for r in system.planets):
        pass  # for_system already picked a published epoch
    else:
        controls.seek(DEFAULT_EPOCHS.get(args.host, 2457000.0), rebase=True)
    epoch = controls.epoch_jd
    print("Hosts placed:      {0}".format(len(explorer.targets)))
    skipped = getattr(explorer, "skipped_hosts", [])
    if skipped:
        print(
            "Hosts without a usable distance, therefore not placed: {0}".format(
                ", ".join(skipped)
            )
        )
    print()
    print("\n".join(explorer.describe()))

    print()
    print("\n".join(controls.describe()))

    if detached:
        print()
        print("\n".join(explorer.describe()))
        _print_panels(explorer, system, epoch)
        return 0

    print()
    print("APPROACH")
    print(
        "  {0:>4}  {1:>14}  {2:>10}  {3:>16}  {4}".format(
            "step", "distance (pc)", "view", "camera (local)", "render"
        )
    )
    for index, position in enumerate(explorer.path_to_system(args.frames)):
        explorer.move_to_pc(position)
        frame = explorer.active_frame
        local = float(np.linalg.norm(explorer.camera_position.values))
        try:
            explorer.camera_position.to_render()
            status = "ok"
        except Exception as exc:  # pragma: no cover - would be a real failure
            status = "PRECISION LOSS: {0}".format(exc)
        print(
            "  {0:>4}  {1:>14.6f}  {2:>10}  {3:>10.4g} {4:<4}  {5}".format(
                index,
                explorer.distance_to_system_pc(),
                explorer.view.value,
                local,
                frame.unit_label,
                status,
            )
        )

    explorer.enter_system()
    print()
    print("\n".join(explorer.describe()))
    _print_panels(explorer, system, epoch)

    if args.no_render:
        return 0

    try:
        import moderngl  # noqa: F401
    except ImportError:
        print('\nModernGL is not installed; skipping the render (pip install -e ".[render]").')
        return 0

    explorer.move_to_pc([0.0, 0.0, 40.0])
    try:
        written = render_approach(explorer, epoch, Path(args.out), frames=args.frames)
    except Exception as exc:  # pragma: no cover - depends on the GL driver
        print("\nRendering FAILED: {0}".format(exc), file=sys.stderr)
        return 1

    print()
    print("Rendered {0} frame(s) -> {1}".format(len(written), written[0].parent))
    for path in written:
        print("  {0}".format(path.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
