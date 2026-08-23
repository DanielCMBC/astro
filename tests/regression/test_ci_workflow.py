"""The CI definition itself.

The suite is now large enough to function as the project's scientific
specification, so it must not depend on anyone remembering to run it. These
tests guard the workflow against the ways a green tick can lie:

* an OpenGL job that silently skips every GL test when the runner has no
  driver, and reports success;
* a workflow that never triggers on the branch being worked on;
* a Python version the package does not actually claim to support.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

pytestmark = pytest.mark.skipif(not WORKFLOW.exists(), reason="CI workflow not present")


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _steps(workflow: dict, job: str) -> list[dict]:
    return workflow["jobs"][job]["steps"]


def _run_text(workflow: dict, job: str) -> str:
    return "\n".join(step.get("run", "") for step in _steps(workflow, job))


def test_the_workflow_parses():
    assert yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_it_triggers_on_every_branch(workflow):
    """Including 3D-test, which is where the work happens."""
    # PyYAML reads a bare `on:` key as the boolean True.
    triggers = workflow.get("on", workflow.get(True))
    assert triggers is not None
    assert "**" in triggers["push"]["branches"]


def test_both_jobs_exist(workflow):
    assert {"tests", "opengl"} <= set(workflow["jobs"])


def test_the_test_matrix_matches_the_supported_pythons(workflow):
    import tomllib

    versions = workflow["jobs"]["tests"]["strategy"]["matrix"]["python-version"]
    assert "3.11" in versions

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    floor = metadata["project"]["requires-python"].lstrip(">=")
    assert min(versions, key=lambda v: tuple(map(int, v.split(".")))) == floor


def test_the_architecture_tests_run_before_the_rest(workflow):
    """A golden-rule failure means the rest is testing the wrong shape."""
    names = [step.get("name", "") for step in _steps(workflow, "tests")]
    architecture = next(i for i, n in enumerate(names) if "Architecture" in n)
    full = next(i for i, n in enumerate(names) if "Full test suite" in n)
    assert architecture < full


def test_the_full_suite_is_run(workflow):
    assert "pytest -q" in _run_text(workflow, "tests")


def test_the_offline_guarantee_is_checked(workflow):
    text = _run_text(workflow, "tests")
    assert "load_reference_catalog" in text
    assert "Offline" in "".join(s.get("name", "") for s in _steps(workflow, "tests"))


# -- the OpenGL job may not report a silent skip ----------------------------


def test_the_opengl_job_verifies_a_context_before_running_tests(workflow):
    """The hole this closes: importorskip plus a skipping fixture is green."""
    names = [step.get("name", "") for step in _steps(workflow, "opengl")]
    verify = next(i for i, n in enumerate(names) if "Verify a GL" in n)
    tests = next(i for i, n in enumerate(names) if "OpenGL backend tests" in n)
    assert verify < tests

    assert "scripts/verify_gl.py" in _run_text(workflow, "opengl")


def _code_only(path: Path) -> str:
    """Source with comments and string literals removed.

    The verifier's own docstring explains what skipping would cost, so
    scanning raw text would flag the explanation as the offence.
    """
    import tokenize

    kept = []
    with open(path, "rb") as handle:
        for token in tokenize.tokenize(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            kept.append(token.string)
    return " ".join(kept)


def test_the_verifier_hard_fails_rather_than_skipping():
    """It must raise, not skip, when no context exists."""
    code = _code_only(ROOT / "scripts" / "verify_gl.py")
    assert "SystemExit" in code
    assert "skip" not in code.lower()
    assert "pytest" not in code.lower()


def test_the_verifier_and_the_test_fixture_share_one_context_factory():
    """Otherwise verification could pass on a backend the tests never try."""
    verifier = (ROOT / "scripts" / "verify_gl.py").read_text(encoding="utf-8")
    fixture = (ROOT / "tests" / "regression" / "test_gl_backend.py").read_text(
        encoding="utf-8"
    )
    for source in (verifier, fixture):
        assert "create_standalone_context" in source
    # Neither may call ModernGL directly any more.
    assert "moderngl.create_standalone_context" not in verifier
    assert "moderngl.create_standalone_context" not in fixture


def test_software_rendering_is_forced_for_the_gl_job(workflow):
    env = workflow["jobs"]["opengl"]["env"]
    assert env["LIBGL_ALWAYS_SOFTWARE"] in ("1", 1, True)
    assert env["MESA_GL_VERSION_OVERRIDE"] == "3.3"
    assert env["MESA_GLSL_VERSION_OVERRIDE"] == "330"


def test_mesa_and_egl_are_installed(workflow):
    text = _run_text(workflow, "opengl")
    for package in ("libgl1-mesa-dri", "libegl1", "xvfb"):
        assert package in text


def test_the_gl_job_installs_the_render_extra(workflow):
    assert '".[dev,render]"' in _run_text(workflow, "opengl")


def test_the_gl_job_renders_every_demo(workflow):
    text = _run_text(workflow, "opengl")
    for demo in ("slice_demo", "system_demo", "explorer_demo", "orientation_demo"):
        assert demo in text


def test_the_gl_job_renders_the_orientation_provenance_cases(workflow):
    """C2's claim is visual, so software Mesa has to draw it.

    Measured, derived and assumed orientations must reach real pixels in
    CI. The difference between them is a dash pattern, which is exactly the
    kind of thing that can be correct in a scene description and lost on
    the way to the GPU.
    """
    text = _run_text(workflow, "opengl")
    assert "orientation_demo" in text
    assert "test_explorer_c2.py" in text


def test_the_gl_job_runs_the_explorer_tests(workflow):
    """Navigation and legacy isolation belong in the GL job too."""
    text = _run_text(workflow, "opengl")
    assert "test_explorer.py" in text
    assert "test_legacy_isolation.py" in text


# -- the context factory itself ---------------------------------------------


def test_the_factory_tries_headless_backends():
    from astro_explorer.rendering.gl_backend import GL_BACKENDS

    assert "egl" in GL_BACKENDS


def test_the_factory_reports_which_backend_it_used():
    """So a software renderer is visible in the CI log, not hidden."""
    pytest.importorskip("moderngl")
    from astro_explorer.rendering.gl_backend import create_standalone_context

    try:
        context, backend = create_standalone_context(require=330)
    except RuntimeError:
        pytest.skip("no OpenGL 3.3 core context available")
    try:
        assert isinstance(backend, str) and backend
        assert context.version_code >= 330
    finally:
        context.release()


def test_pillow_is_declared_because_labels_need_it():
    """rendering/labels.py imports PIL; it must not ride in on matplotlib."""
    import tomllib

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    render = " ".join(metadata["project"]["optional-dependencies"]["render"]).lower()
    assert "pillow" in render


# -- the two defects the first green CI run was hiding -----------------------


def test_the_workflow_asserts_the_frames_were_produced(workflow):
    """The first run passed while rendering nothing at all.

    Both demos caught the exception, printed a note and returned 0, so the
    step went green with an empty output directory. The upload step's
    "no files found" was only a warning.
    """
    names = [step.get("name", "") for step in _steps(workflow, "opengl")]
    assert any("Check the frames were actually produced" in n for n in names)

    check = next(
        s for s in _steps(workflow, "opengl")
        if "Check the frames" in s.get("name", "")
    )
    assert "exit 1" in check["run"]

    render = next(i for i, n in enumerate(names) if n.startswith("Render the"))
    verify = next(i for i, n in enumerate(names) if "Check the frames" in n)
    assert render < verify


@pytest.mark.parametrize("module", ["slice_demo", "system_demo"])
def test_a_failed_render_exits_non_zero(module, monkeypatch, tmp_path):
    """A render that was asked for and failed is an error, not a shrug."""
    import importlib

    demo = importlib.import_module("astro_explorer.app.{0}".format(module))
    target = "render_phases" if module == "slice_demo" else "render_system"

    def boom(*args, **kwargs):
        raise RuntimeError("the number of samples is invalid")

    monkeypatch.setattr(demo, target, boom)
    pytest.importorskip("moderngl")

    host = "HD 80606" if module == "slice_demo" else "Kepler-11"
    code = demo.main(["--host", host, "--out", str(tmp_path)])
    assert code == 1


@pytest.mark.parametrize("module", ["slice_demo", "system_demo"])
def test_a_missing_renderer_is_not_an_error(module, monkeypatch, tmp_path):
    """Not having ModernGL installed is benign; a broken render is not."""
    import builtins
    import importlib

    demo = importlib.import_module("astro_explorer.app.{0}".format(module))
    real_import = builtins.__import__

    def no_moderngl(name, *args, **kwargs):
        if name == "moderngl":
            raise ImportError("no moderngl")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_moderngl)
    host = "HD 80606" if module == "slice_demo" else "Kepler-11"
    assert demo.main(["--host", host, "--out", str(tmp_path)]) == 0


def test_multisampling_degrades_instead_of_failing():
    """llvmpipe rejected 8x MSAA, which is what broke the first CI render."""
    pytest.importorskip("moderngl")
    from astro_explorer.rendering.gl_backend import GLRenderer, RenderSettings

    try:
        renderer = GLRenderer(RenderSettings(width=64, height=64, samples=4096))
    except RuntimeError:
        pytest.skip("no OpenGL 3.3 core context available")
    try:
        # Whatever the context supports, it must not have simply used 4096.
        assert renderer.samples <= max(int(renderer.ctx.max_samples), 1)
        assert renderer.samples >= 1
    finally:
        renderer.release()


def test_rendering_still_works_at_the_clamped_sample_count():
    pytest.importorskip("moderngl")
    import numpy as np

    from astro_explorer.rendering.camera import Camera
    from astro_explorer.rendering.gl_backend import GLRenderer, RenderSettings
    from astro_explorer.rendering.renderer import RenderStar, SceneDescription

    try:
        renderer = GLRenderer(RenderSettings(width=96, height=96, samples=4096))
    except RuntimeError:
        pytest.skip("no OpenGL 3.3 core context available")
    try:
        scene = SceneDescription(stars=[RenderStar("S", [0, 0, 0], 1.0, (1, 1, 1))])
        camera = Camera(target=np.zeros(3), distance=4.0, aspect=1.0)
        image = renderer.render(scene, camera)
        assert int((image.sum(axis=2) > 24).sum()) > 100
    finally:
        renderer.release()


def test_the_test_job_names_the_coordinate_inspector_step(workflow):
    """C3 is auditable on its own line, not only inside the total.

    The C3 contract - a distance that does not move when the camera does -
    is the kind of thing that is easy to break in an unrelated rendering
    change. A named step means the break is legible on the CI page instead
    of being one failure inside a full-suite run.
    """
    steps = _steps(workflow, "tests")
    named = [s for s in steps if "Explorer C3" in s.get("name", "")]
    assert named, [s.get("name") for s in steps]
    assert "test_explorer_c3.py" in named[0]["run"]

    # And it runs before the full suite, so the specific signal arrives first.
    names = [s.get("name", "") for s in steps]
    assert names.index(named[0]["name"]) < names.index("Full test suite")
