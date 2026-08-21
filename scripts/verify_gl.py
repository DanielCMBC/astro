"""Hard-fail check that a real GL 3.3 core context exists and shaders compile.

The GL tests in ``tests/regression/test_gl_backend.py`` skip when no context
can be created, which is right for a developer laptop without a GPU but
wrong for CI: a misconfigured runner would silently skip every OpenGL test
and still report green.

This script exists so the CI job fails loudly instead. It:

1. creates a standalone GL 3.3 core context, trying each ModernGL backend;
2. prints what it actually got, so a software renderer is visible in the log;
3. compiles every declared shader program;
4. renders one frame offscreen and checks that pixels were written.

Run it directly::

    python scripts/verify_gl.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def create_context():
    """Delegate to the package factory, so CI and the tests agree exactly."""
    from astro_explorer.rendering.gl_backend import create_standalone_context

    try:
        return create_standalone_context(require=330)
    except RuntimeError as exc:
        raise SystemExit(str(exc))


def main() -> int:
    from astro_explorer.rendering.renderer import PROGRAMS, ShaderLibrary

    context, backend = create_context()
    info = context.info
    print("OpenGL context created via backend: {0}".format(backend))
    for key in ("GL_VERSION", "GL_RENDERER", "GL_VENDOR", "GL_SHADING_LANGUAGE_VERSION"):
        print("  {0:<28} {1}".format(key, info.get(key, "?")))

    version = context.version_code
    if version < 330:
        raise SystemExit("context reports {0}, need at least 330".format(version))

    library = ShaderLibrary()
    print("\nCompiling {0} declared shader programs:".format(len(PROGRAMS)))
    for name in sorted(PROGRAMS):
        program = context.program(**library.program_sources(name))
        attributes = sorted(program)
        print("  {0:<12} ok  ({1} active names)".format(name, len(attributes)))
        program.release()

    # A context that compiles shaders but writes no pixels is still broken.
    print("\nRendering one offscreen frame:")
    import numpy as np

    from astro_explorer.rendering.camera import Camera
    from astro_explorer.rendering.gl_backend import GLRenderer, RenderSettings
    from astro_explorer.rendering.renderer import RenderPlanet, RenderStar, SceneDescription

    renderer = GLRenderer(
        RenderSettings(width=160, height=120, samples=1), context=context, lod=2
    )
    scene = SceneDescription(
        stars=[RenderStar("S", [0.0, 0.0, 0.0], 0.6, (1.0, 0.95, 0.85))],
        planets=[RenderPlanet("P", [2.0, 0.0, 0.0], 0.3, base_color=(0.8, 0.6, 0.4))],
    )
    camera = Camera(target=np.zeros(3), distance=6.0, aspect=160 / 120, pitch=0.3)
    image = renderer.render(scene, camera)
    lit = int((image.sum(axis=2) > 24).sum())
    print("  lit pixels: {0}".format(lit))
    renderer.release()

    if lit < 50:
        raise SystemExit("the frame rendered {0} lit pixels; expected many more".format(lit))

    print("\nOpenGL verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
