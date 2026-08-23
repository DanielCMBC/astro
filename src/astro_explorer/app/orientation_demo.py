"""Explorer C2: the orbital-orientation overlay, drawn three ways.

    python -m astro_explorer.app.orientation_demo
    python -m astro_explorer.app.orientation_demo --out artifacts
    python -m astro_explorer.app.orientation_demo --no-render

The overlay's difficulty is not geometry. It is that a line of nodes drawn
from a measured longitude of the ascending node and one drawn from a
display normalisation are *the same line*, and a picture cannot tell them
apart unless it is made to.

So this demo renders the three cases side by side and says which is which:

    MEASURED   i, omega and Omega all published under a stated convention
    DERIVED    omega converted from the host star's reflex orbit, +180 deg
    ASSUMED    omega's convention unstated, Omega never observed at all

The first two are **constructed element sets**, not catalogue systems. They
have to be: no exoplanet in the committed snapshot has a measured longitude
of the ascending node, because it is not observable from transits or radial
velocity. Inventing a star to demonstrate what a measured orientation would
look like is honest as long as it is labelled, and it is labelled in the
header of every frame it appears in.

The third case is real - HD 80606 b, whose argument of periastron is
published with no stated convention - and it is the one that looks like
almost every planet in the archive.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np

from ..coordinates.system_frame import SystemFrame
from ..physics.orbital_elements import OrbitalElements
from ..physics.orbital_semantics import PeriastronConvention
from ..provenance import measured
from ..rendering.renderer import RenderOrbit, RenderStar, SceneDescription
from ..rendering.scene_builder import orbit_path, orientation_guides
from .vertical_slice import build_slice, load_reference_catalog

__all__ = ["main", "Case", "cases", "build_orientation_scene", "render_cases"]


@dataclass
class Case:
    """One orientation example: elements, a name, and what it demonstrates."""

    key: str
    title: str
    elements: OrbitalElements
    #: True when the elements were constructed for the demonstration rather
    #: than read from the catalogue. Always stated on the frame.
    synthetic: bool = True
    #: Whether guides for unpublished elements are drawn at their display
    #: normalisation. On for the assumed case, which is what it is showing.
    show_normalised: bool = False


def _angle(degrees: float):
    return measured(
        float(degrees), u.deg, provenance="constructed demonstration"
    ).to(u.rad)


def _constructed(convention: PeriastronConvention) -> OrbitalElements:
    """An element set with every orientation angle published.

    The numbers are arbitrary and the provenance says so: what the frame is
    demonstrating is the *style* a fully determined orientation is drawn
    in, which no real exoplanet in the snapshot can show.
    """
    return OrbitalElements(
        name="constructed example b",
        semimajor_axis=measured(1.0, u.au, provenance="constructed demonstration"),
        eccentricity=measured(0.45, provenance="constructed demonstration"),
        period=measured(365.0, u.day, provenance="constructed demonstration"),
        inclination=_angle(38.0),
        argument_of_periastron=_angle(55.0),
        longitude_of_ascending_node=_angle(120.0),
        periastron_convention=convention,
        reference="constructed for the C2 demonstration; not a catalogue system",
    )


def cases(catalog=None) -> list[Case]:
    """The three orientation cases, in increasing order of doubt."""
    found = [
        Case(
            key="measured",
            title="MEASURED - i, omega and Omega published, convention stated",
            elements=_constructed(PeriastronConvention.PLANET),
        ),
        Case(
            key="derived",
            title="DERIVED - omega converted from the star's reflex orbit (+180 deg)",
            elements=_constructed(PeriastronConvention.STELLAR_REFLEX),
        ),
    ]

    if catalog is not None:
        system = build_slice("HD 80606", catalog)
        record = system.planet("HD 80606 b")
        found.append(
            Case(
                key="assumed",
                title="ASSUMED - HD 80606 b: omega's convention unstated, Omega never observed",
                elements=record.elements,
                synthetic=False,
                show_normalised=True,
            )
        )
    return found


def build_orientation_scene(case: Case, frame: SystemFrame) -> SceneDescription:
    """A star, one orbit, and the orientation guides for it.

    Deliberately sparse: this frame is about how the orbit sits in space,
    so anything else on screen is a distraction from the one thing being
    demonstrated.
    """
    scene = SceneDescription(unit_label=frame.unit_label)
    scene.stars.append(
        RenderStar(
            identifier="{0}:star".format(case.key),
            position_local=frame.star_position().to_render(),
            radius_display=0.06,
            color=(1.0, 0.94, 0.82),
            label="host",
        )
    )

    display = case.elements.for_display()
    path = orbit_path(display, samples=720)
    if path is not None:
        scene.orbits.append(
            RenderOrbit(
                identifier="{0}:orbit".format(case.key),
                points_local=frame.place_planet(path).to_render(),
                dashed=False,
            )
        )

    overlay = orientation_guides(
        case.elements,
        frame,
        identifier=case.key,
        show_normalised=case.show_normalised,
        label=case.elements.name,
    )
    scene.guides.extend(overlay.guides)

    scene.annotations.append(case.title)
    if case.synthetic:
        scene.annotations.append(
            "Constructed element set, not a catalogue system: no exoplanet in "
            "the snapshot has a measured longitude of the ascending node."
        )
    scene.annotations.extend(overlay.annotations)
    return scene


def render_cases(found: list[Case], out_dir: Path, *, width: int = 1400, height: int = 900):
    """Render one frame per case, with its provenance printed on it."""
    from ..rendering.camera import Camera
    from ..rendering.gl_backend import GLRenderer, RenderSettings
    from ..rendering.labels import LabelStyle, draw_labels

    out_dir.mkdir(parents=True, exist_ok=True)
    settings = RenderSettings(width=width, height=height, samples=8)

    written = []
    with GLRenderer(settings) as renderer:
        for case in found:
            frame = SystemFrame.for_host(case.key)
            scene = build_orientation_scene(case, frame)

            camera = Camera(target=np.zeros(3), aspect=width / height, pitch=0.85, yaw=0.5)
            camera.frame_object(scene.bounding_radius(), margin=1.3)
            image = renderer.render(scene, camera)

            header = [case.title]
            header += [
                note
                for note in scene.annotations[1:]
                if note.startswith(("Ascending node", "Argument of periastron", "Inclination"))
            ]
            image = draw_labels(
                image,
                scene.project_labels(camera, width, height),
                LabelStyle(),
                header=header[:5],
            )

            path = out_dir / "orientation_{0}.png".format(case.key)
            renderer.save_png(image, path)
            written.append(path)

        guide_draw_calls = renderer.last_guide_draw_calls
    return written, guide_draw_calls


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="renders")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args(argv)

    try:
        catalog = load_reference_catalog()
    except FileNotFoundError:
        catalog = None
        print("reference snapshot not found; the real ASSUMED case is skipped\n")

    found = cases(catalog)

    for case in found:
        frame = SystemFrame.for_host(case.key)
        scene = build_orientation_scene(case, frame)
        print("=" * 78)
        print(case.title)
        print("-" * 78)
        for note in scene.annotations[1:]:
            print("  " + note)
        print("  guides drawn: {0}".format(len(scene.guides)))
        for guide in scene.guides:
            print(
                "    {0:<22} {1}".format(
                    guide.identifier.rsplit(":", 1)[-1], guide.style.value.lower()
                )
            )
        print()

    if args.no_render:
        return 0

    try:
        import moderngl  # noqa: F401
    except ImportError:
        print('ModernGL is not installed; skipping the render (pip install -e ".[render]").')
        return 0

    try:
        written, draw_calls = render_cases(found, Path(args.out))
    except Exception as exc:  # pragma: no cover - depends on the GL driver
        # Rendering was asked for and failed. Returning 0 here would let CI
        # report success having produced no image at all.
        print("\nRendering FAILED: {0}".format(exc), file=sys.stderr)
        return 1

    print("Rendered {0} frame(s) -> {1}".format(len(written), written[0].parent))
    print("  every guide in the last frame cost {0} draw call(s)".format(draw_calls))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
