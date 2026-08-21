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


def test_the_gl_job_renders_both_demos(workflow):
    text = _run_text(workflow, "opengl")
    assert "slice_demo" in text
    assert "system_demo" in text


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
