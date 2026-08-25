"""Explorer C3.6: astrometric epoch and stellar space motion.

C3.5 shipped the SystemFrame -> ICRS rotation and then declined to publish
an absolute planet position, gating it on a boolean named ``epoch_resolved``
that defaulted to False and was set True only inside tests. That boolean was
the last deliberately closed gate in the pipeline, and it had the shape of a
gate that opens by accident: one ``True`` written at one call site would have
unlocked every downstream position without a single number changing.

This slice replaces it with the state it stood in for - a reference epoch,
proper motion, radial velocity, and a propagation to a concrete instant -
and these tests are mostly about the ways that can go quietly wrong.

Every failure mode below is silent
----------------------------------

None of these produce an exception, a NaN, or a visibly odd picture:

* **a missing radial velocity read as zero.** Astropy's
  ``apply_space_motion`` substitutes zero for any differential a coordinate
  does not carry, so a star with proper motion and no RV propagates to a
  perfectly plausible, perfectly confident, wrong distance;
* **``pmra`` multiplied by ``cos(dec)``.** Gaia's ``pmra`` is already
  :math:`\\mu_\\alpha^*`. Applying the factor again is exact at the equator
  and halves the motion at 60 degrees - a bug whose severity depends on
  which star you happen to test;
* **a reference epoch inferred rather than stored.** Assuming J2000 for a
  Gaia position is sixteen years of proper motion: 33 arcsec for HD 219134;
* **two objects propagated to different instants.** Each state is
  impeccable; their difference includes both stars' motion over the gap;
* **a tangent basis built from the catalogue RA/Dec while the origin is the
  propagated one.** The offset then lands in a basis rotated away from the
  position it is added to.

So the checks here compare against Astropy computed independently, against
values that must *not* change, and against the specific wrong answers.

Fixtures
--------

Four kinds, because the tiers only separate under all of them:

* a synthetic high-proper-motion star, where a missed propagation is
  enormous rather than subtle;
* a synthetic star with *explicitly measured* zero motion - the case that
  must survive as zero while a missing one must not become zero;
* real Gaia DR3-backed exoplanet hosts from the committed cache, including
  TRAPPIST-1, which genuinely has no Gaia radial velocity and is therefore a
  real direction-only case rather than a contrived one;
* the existing detached/unlocated regression case.
"""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from astro_explorer.app.vertical_slice import build_slice, load_reference_catalog
from astro_explorer.coordinates.astrometry import (
    DIRECTION_ONLY_IS_MODEL_DEPENDENT,
    DIRECTION_ONLY_NOT_A_POSITION,
    SPACE_MOTION_MODEL,
    ZERO_RV_APPROXIMATION,
    PROPER_MOTION_NOT_MEASURED,
    RADIAL_VELOCITY_NOT_MEASURED,
    REFERENCE_EPOCH_NOT_STATED,
    AstrometricState,
    MotionKnowledge,
    at_reference_epoch,
    propagate_astrometry,
)
from astro_explorer.coordinates.frames import Frame, sky_position
from astro_explorer.coordinates.inspector import (
    COMMON_TIME_VIOLATED,
    PLANET_PHASE_NOT_CONSTRAINED,
    PLANET_STATE_NOT_TIMED,
    PLANET_TIME_MISMATCH,
    TARGET_ASTROMETRY_NOT_PROPAGATED,
    absolute_planet_position,
    planet_to_star_distance,
)
from astro_explorer.coordinates.tangent import (
    ASTROMETRY_NOT_PROPAGATED,
    node_sense_of,
    NODE_CONVENTION_UNSTATED,
    NODE_SENSE_UNRESOLVED,
    NodeConvention,
    NodeSense,
    NodeSenseEvidence,
    absolute_position_blockers,
    system_offset_to_icrs_pc,
    tangent_basis,
)
from astro_explorer.data.gaia import (
    CACHE_SCHEMA_VERSION,
    GAIA_DR3_RELEASE,
    GAIA_DR3_REFERENCE_EPOCH_JYEAR,
    GAIA_DR3_TIME_SCALE,
    GaiaAstrometryCache,
    GaiaAstrometryRecord,
    GaiaHostIndex,
    astrometric_state_from_gaia,
    gaia_dr3_ids_from_catalog,
    parse_gaia_dr3_id,
    validate_gaia_record,
)
from astro_explorer.physics.epoch import TimeScale, astropy_time, orbital_time_jd
from astro_explorer.physics.orbital_elements import OrbitalElements
from astro_explorer.physics.orbital_semantics import (
    CIRCULAR_ORBIT_PERIAPSIS_NOTE,
    INCLINATION_UNRESOLVED,
    PERIAPSIS_DIRECTION_UNRESOLVED,
    PERIASTRON_CONVENTION_UNSTATED,
    PeriastronConvention,
    absolute_orientation_blockers,
)
from astro_explorer.physics.phase import PhaseProvenance, PhaseSolution
from astro_explorer.physics.state_vectors import StateVector
from astro_explorer.physics.timed_state import TimedOrbitalState, same_instant
from astro_explorer.physics.node_semantics import (
    NODE_CONVENTION_NOT_RENDERABLE,
    resolve_node_azimuth,
    resolve_node_azimuth_detailed,
)
from astro_explorer.provenance import Status, assumed, measured, unknown

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "astro_explorer"


def _code_only(path: Path) -> str:
    """Source with comments and string literals removed."""
    import tokenize

    kept = []
    with open(path, "rb") as handle:
        for token in tokenize.tokenize(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            kept.append(token.string)
    return " ".join(kept)


EPOCH_JD = 2460000.0
J2000 = astropy_time(2451545.0)
GAIA_EPOCH = Time(GAIA_DR3_REFERENCE_EPOCH_JYEAR, format="jyear", scale=GAIA_DR3_TIME_SCALE)


# ==========================================================================
# Fixtures
# ==========================================================================


@pytest.fixture(scope="module")
def catalog():
    try:
        return load_reference_catalog()
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("reference snapshot not committed")


@pytest.fixture(scope="module")
def gaia_cache() -> GaiaAstrometryCache:
    index = GaiaHostIndex()
    if not index.cache.exists:  # pragma: no cover
        pytest.skip("Gaia astrometry cache not committed")
    return index.cache


@pytest.fixture(scope="module")
def host_index() -> GaiaHostIndex:
    index = GaiaHostIndex()
    if not index.hosts:  # pragma: no cover
        pytest.skip("Gaia host map not committed")
    return index


@pytest.fixture(scope="module")
def hd80606(catalog):
    """Gaia-backed, full five-parameter solution plus a radial velocity."""
    return build_slice("HD 80606", catalog)


@pytest.fixture(scope="module")
def hd219134(catalog):
    """Gaia-backed, and the fastest-moving host in the snapshot."""
    return build_slice("HD 219134", catalog)


@pytest.fixture(scope="module")
def trappist1(catalog):
    """Real direction-only case: Gaia publishes no radial velocity for it."""
    return build_slice("TRAPPIST-1", catalog)


def _high_proper_motion() -> AstrometricState:
    """A synthetic star moving 10 arcsec/yr - Barnard's-Star scale.

    Synthetic on purpose: the numbers are round so that a propagation over a
    known interval has an arithmetic expectation, and fast so that a
    mishandled epoch is a gross error rather than a marginal one.
    """
    return AstrometricState(
        position=sky_position(
            "fast probe", 100.0, 20.0, parallax_mas=measured(500.0, u.mas)
        ),
        source_catalog="synthetic",
        source_id="fast",
        reference_epoch=J2000,
        pm_ra_cosdec=measured(-8000.0, u.mas / u.yr, provenance="synthetic"),
        pm_dec=measured(6000.0, u.mas / u.yr, provenance="synthetic"),
        radial_velocity=measured(-110.0, u.km / u.s, provenance="synthetic"),
    )


def _measured_zero_motion() -> AstrometricState:
    """A star whose motion was *measured* and found to be zero.

    Not the same object as a star with no published motion, and the whole
    reason :class:`~astro_explorer.provenance.Status` travels on every
    component. A survey that measures 0.00 +/- 0.02 mas/yr has said
    something; a catalogue with an empty column has not.
    """
    return AstrometricState(
        position=sky_position(
            "still probe", 210.0, -33.0, parallax_mas=measured(4.0, u.mas)
        ),
        source_catalog="synthetic",
        source_id="still",
        reference_epoch=J2000,
        pm_ra_cosdec=measured(
            0.0, u.mas / u.yr, error_plus=0.02, error_minus=0.02, provenance="synthetic"
        ),
        pm_dec=measured(
            0.0, u.mas / u.yr, error_plus=0.02, error_minus=0.02, provenance="synthetic"
        ),
        radial_velocity=measured(
            0.0, u.km / u.s, error_plus=0.4, error_minus=0.4, provenance="synthetic"
        ),
    )


def _no_motion_published() -> AstrometricState:
    """The same star with the motion columns simply absent."""
    return AstrometricState(
        position=sky_position(
            "silent probe", 210.0, -33.0, parallax_mas=measured(4.0, u.mas)
        ),
        source_catalog="synthetic",
        source_id="silent",
        reference_epoch=J2000,
    )


def _oriented_elements(
    *,
    inclination_deg=63.0,
    omega_deg=41.0,
    node=None,
    convention=PeriastronConvention.PLANET,
    eccentricity=0.2,
):
    """An element set whose physical orientation is fully observed.

    C3.6's second audit added the gate this satisfies: a unique absolute
    position needs all three Euler angles to be observations, not just the
    node. The defaults here are the *passing* case, so a test that is about
    something else - a clock, a basis, an astrometric tier - is not
    accidentally testing the orientation gate as well. Tests that are about
    orientation deliberately weaken one angle at a time.
    """
    return OrbitalElements(
        name="probe b",
        semimajor_axis=measured(1.0, u.au, provenance="test"),
        eccentricity=measured(eccentricity, provenance="test"),
        period=measured(365.0, u.day, provenance="test"),
        inclination=(
            unknown(u.rad)
            if inclination_deg is None
            else measured(np.deg2rad(inclination_deg), u.rad, provenance="test")
        ),
        argument_of_periastron=(
            unknown(u.rad)
            if omega_deg is None
            else measured(np.deg2rad(omega_deg), u.rad, provenance="test")
        ),
        longitude_of_ascending_node=_resolved_node() if node is None else node,
        periastron_convention=convention,
    )


def _timed(
    offset_au, *, obstime=None, phase=None, elements=None
) -> TimedOrbitalState:
    """A planet offset that carries the instant it holds at, and its orbit.

    The defaults are the passing case in both dimensions: a published
    periastron epoch, so the *date* is a real observation, and a fully
    observed orientation, so a test about clocks is not silently also a test
    about Euler angles.
    """
    return TimedOrbitalState(
        state=StateVector(position=np.asarray(offset_au, dtype=np.float64)),
        obstime=obstime if obstime is not None else astropy_time(EPOCH_JD),
        phase=phase or PhaseSolution(0.0, PhaseProvenance.PERIASTRON_EPOCH),
        elements=elements if elements is not None else _oriented_elements(),
    )


def _resolved_node(value_rad: float = 0.7):
    """A node that clears every node gate, so only astrometry is under test."""
    node = measured(value_rad, u.rad, provenance="test: RV-resolved")
    return replace(
        node,
        extra={
            "node_convention": NodeConvention.PA_EAST_OF_NORTH_RECEDING.value,
            "node_sense": NodeSense.RESOLVED.value,
            "node_sense_evidence": NodeSenseEvidence.ORBITAL_RV_SOLUTION.value,
        },
    )


# ==========================================================================
# The Gaia identity: deterministic, and never a cone search
# ==========================================================================


def test_a_gaia_dr3_id_maps_deterministically_to_a_source_id():
    """The archive already made the cross-match; this only reads it.

    Determinism is the property being claimed: the same input always yields
    the same source, with no positional tolerance, no epoch, and no nearest
    neighbour involved. Re-deriving the match by cone search would be least
    reliable exactly where it matters most - a crowded field around a
    high-proper-motion star whose epoch we cannot state.
    """
    assert parse_gaia_dr3_id("Gaia DR3 2009481748875806976") == "2009481748875806976"
    assert parse_gaia_dr3_id("gaia dr3 2009481748875806976") == "2009481748875806976"
    assert parse_gaia_dr3_id("2009481748875806976") == "2009481748875806976"
    assert parse_gaia_dr3_id(" Gaia DR3 2009481748875806976 ") == "2009481748875806976"

    # Repeated calls, and the whole of a catalogue column, agree.
    for _ in range(3):
        assert parse_gaia_dr3_id("Gaia DR3 4062446910648807168") == "4062446910648807168"

    for absent in (None, "", "  ", "null", "TYC 1234-5678-1", "Gaia DR2"):
        assert parse_gaia_dr3_id(absent) is None


def test_the_identifier_is_a_string_because_it_does_not_survive_a_float():
    """A DR3 source_id is a 64-bit integer and many exceed 2^53.

    Round-tripping one through a float - which happens the moment it meets a
    numeric pandas column or a JSON reader that prefers numbers - changes
    which star it names, and changes it to a real neighbouring source rather
    than to something obviously broken.
    """
    source_id = parse_gaia_dr3_id("Gaia DR3 2635476908753563008")  # TRAPPIST-1
    assert isinstance(source_id, str)
    assert int(source_id) > 2**53

    # This one really does change identity through a float, and it changes
    # into another valid-looking 19-digit id rather than into nonsense.
    through_a_float = str(int(float(source_id)))
    assert through_a_float != source_id
    assert len(through_a_float) == len(source_id)


def test_two_identifiers_for_one_host_is_a_refusal_not_a_choice():
    """A cross-match disagreement must not be resolved by row order."""
    rows = [
        {"hostname": "Probe", "gaia_dr3_id": "Gaia DR3 1111111111111111111"},
        {"hostname": "Probe", "gaia_dr3_id": "Gaia DR3 2222222222222222222"},
    ]
    with pytest.raises(ValueError, match="two Gaia DR3 identifiers"):
        gaia_dr3_ids_from_catalog(rows)


def test_one_identifier_repeated_across_planets_is_one_host(host_index):
    """TRAPPIST-1's seven rows carry one source; the index holds one entry."""
    rows = [
        {"hostname": "TRAPPIST-1", "gaia_dr3_id": "Gaia DR3 2635476908753563008"}
        for _ in range(7)
    ]
    assert gaia_dr3_ids_from_catalog(rows) == {
        "TRAPPIST-1": "2635476908753563008"
    }
    assert host_index.source_id_for("TRAPPIST-1") == "2635476908753563008"


# ==========================================================================
# What Gaia said, stored rather than assumed
# ==========================================================================


def test_the_gaia_reference_epoch_is_stored_not_inferred(gaia_cache):
    """``ref_epoch`` is a column, and the code reads the column.

    Hard-coding J2016.0 would work for DR3 and keep working - silently and
    wrongly - into whatever release changes it. So the record carries what
    the archive returned, the cache stores it, and the documented value is
    used only to *reject* a row that disagrees.
    """
    records = gaia_cache.records()
    assert records, "the committed cache should not be empty"

    for source_id, record in records.items():
        assert record.ref_epoch is not None, source_id
        assert record.ref_epoch == pytest.approx(GAIA_DR3_REFERENCE_EPOCH_JYEAR)
        state = astrometric_state_from_gaia(record)
        assert state.reference_epoch is not None
        assert state.reference_epoch.scale == GAIA_DR3_TIME_SCALE
        assert state.reference_epoch_jyear == pytest.approx(
            GAIA_DR3_REFERENCE_EPOCH_JYEAR
        )

    # And a row whose epoch disagrees is rejected rather than corrected.
    drifted = replace(next(iter(records.values())), ref_epoch=2000.0)
    errors = validate_gaia_record(drifted)
    assert any("ref_epoch" in e for e in errors)

    absent = replace(next(iter(records.values())), ref_epoch=None)
    assert any("stored, not inferred" in e for e in validate_gaia_record(absent))


def test_gaia_pmra_maps_directly_to_pm_ra_cosdec(gaia_cache):
    """``pmra`` is already mu_alpha* = d(alpha)/dt cos(delta).

    So the mapping is an assignment. The tempting ``* cos(dec)`` is exact at
    the equator and wrong by a factor of two at 60 degrees, which means a
    test written against one low-declination star would pass while the bug
    was present.

    HD 219134 sits at +57 degrees, where ``cos(dec)`` is 0.54 - far enough
    from 1 that a double application cannot hide.
    """
    record = gaia_cache.record_for("2009481748875806976")
    assert record is not None
    state = astrometric_state_from_gaia(record, name="HD 219134")

    assert state.pm_ra_cosdec.value_in(u.mas / u.yr) == pytest.approx(
        record.pmra, rel=1e-15
    )
    assert state.pm_dec.value_in(u.mas / u.yr) == pytest.approx(record.pmdec, rel=1e-15)

    # The specific wrong answer, named so it cannot be reintroduced quietly.
    cos_dec = np.cos(np.deg2rad(record.dec))
    assert cos_dec < 0.6
    assert state.pm_ra_cosdec.value_in(u.mas / u.yr) != pytest.approx(
        record.pmra * cos_dec, rel=1e-6
    )

    # And the convention is written into the parameter, not only the docs.
    assert "cos(delta)" in state.pm_ra_cosdec.note


def test_a_missing_motion_component_stays_unknown_and_never_becomes_zero(gaia_cache):
    """TRAPPIST-1 has no Gaia radial velocity. It does not have one of zero.

    This is a real catalogue case, not a contrived one: Gaia publishes an RV
    for a minority of sources. Reading the empty cell as 0 km/s would give
    the star a confident distance at every epoch except the reference one.
    """
    record = gaia_cache.record_for("2635476908753563008")
    assert record is not None
    assert record.radial_velocity is None

    state = astrometric_state_from_gaia(record, name="TRAPPIST-1")
    assert not state.radial_velocity.is_known
    assert state.radial_velocity.value is None
    assert state.radial_velocity.status is Status.UNKNOWN
    assert not state.has_radial_velocity
    assert state.knowledge is MotionKnowledge.DIRECTION_ONLY

    # It has proper motion, so the tier above bare position is reached ...
    assert state.has_proper_motion
    # ... and the tier that needs an RV is not.
    assert RADIAL_VELOCITY_NOT_MEASURED in state.blockers_for(
        MotionKnowledge.FULL_SPACE_MOTION
    )


def test_an_absent_proper_motion_pair_stays_unknown():
    """No motion published is not motion measured to be zero."""
    state = _no_motion_published()

    for parameter in (state.pm_ra_cosdec, state.pm_dec, state.radial_velocity):
        assert not parameter.is_known
        assert parameter.value is None
        assert parameter.status is Status.UNKNOWN

    assert state.knowledge is MotionKnowledge.REFERENCE_EPOCH_ONLY
    assert PROPER_MOTION_NOT_MEASURED in state.blockers_for(MotionKnowledge.DIRECTION_ONLY)


def test_one_proper_motion_component_without_the_other_is_rejected():
    """Half a motion is a motion at the wrong position angle.

    And it looks exactly like a motion, which is why the pair is required
    rather than each component defaulting independently.
    """
    half = GaiaAstrometryRecord(
        source_id="1234567890123456789",
        ref_epoch=GAIA_DR3_REFERENCE_EPOCH_JYEAR,
        ra=10.0,
        dec=20.0,
        pmra=100.0,
        pmdec=None,
        astrometric_params_solved=31,
    )
    assert any("a pair or neither" in e for e in validate_gaia_record(half))

    state = AstrometricState(
        position=sky_position("half", 10.0, 20.0, parallax_mas=measured(10.0, u.mas)),
        reference_epoch=J2000,
        pm_ra_cosdec=measured(100.0, u.mas / u.yr, provenance="test"),
    )
    assert not state.has_proper_motion
    assert state.knowledge is MotionKnowledge.REFERENCE_EPOCH_ONLY


def test_an_assumed_motion_does_not_count_as_a_measured_one():
    """A value invented for visualisation must not open a science gate."""
    state = AstrometricState(
        position=sky_position("probe", 10.0, 20.0, parallax_mas=measured(10.0, u.mas)),
        reference_epoch=J2000,
        pm_ra_cosdec=assumed(0.0, u.mas / u.yr, provenance="display-normalisation"),
        pm_dec=assumed(0.0, u.mas / u.yr, provenance="display-normalisation"),
        radial_velocity=assumed(0.0, u.km / u.s, provenance="display-normalisation"),
    )
    assert not state.has_proper_motion
    assert not state.has_radial_velocity
    assert state.knowledge is MotionKnowledge.REFERENCE_EPOCH_ONLY


def test_a_proper_motion_present_without_a_fitted_solution_is_rejected():
    """``astrometric_params_solved`` has to agree with the columns."""
    inconsistent = GaiaAstrometryRecord(
        source_id="1234567890123456789",
        ref_epoch=GAIA_DR3_REFERENCE_EPOCH_JYEAR,
        ra=10.0,
        dec=20.0,
        pmra=100.0,
        pmdec=50.0,
        astrometric_params_solved=3,  # a two-parameter, position-only source
    )
    assert any(
        "did not fit one" in e for e in validate_gaia_record(inconsistent)
    )


# ==========================================================================
# Explicitly measured zero motion
# ==========================================================================


def test_an_explicitly_measured_zero_motion_remains_zero():
    """Zero is a measurement, and it propagates to itself.

    The pair of rules only works if both halves hold: a missing component
    must not become zero, *and* a measured zero must not be mistaken for a
    missing one. If ``is_known`` were implemented as truthiness anywhere,
    this is the test that catches it.
    """
    state = _measured_zero_motion()

    assert state.pm_ra_cosdec.is_known
    assert state.pm_ra_cosdec.value == 0.0
    assert state.has_proper_motion
    assert state.has_radial_velocity
    assert state.knowledge is MotionKnowledge.FULL_SPACE_MOTION

    far = astropy_time(2500000.0)  # over a century past J2000
    moved = propagate_astrometry(state, far)
    assert moved is not None
    assert moved.is_publishable
    assert moved.blockers == ()

    assert moved.position.ra.value_in(u.deg) == pytest.approx(
        state.position.ra.value_in(u.deg), abs=1e-12
    )
    assert moved.position.dec.value_in(u.deg) == pytest.approx(
        state.position.dec.value_in(u.deg), abs=1e-12
    )
    assert moved.position.distance.value_in(u.pc) == pytest.approx(
        state.position.distance.value_in(u.pc), rel=1e-12
    )


# ==========================================================================
# The reference epoch: the answer that needs no model
# ==========================================================================


def test_a_reference_epoch_position_needs_no_invented_motion():
    """At its own epoch, a dated position is complete.

    A star with no measured motion at all is fully publishable *there*, and
    withholding it for lack of a proper motion would refuse the one answer
    that requires no model. The knowledge tiers exist so this case is a
    result rather than a degraded version of one.
    """
    state = _no_motion_published()
    assert state.knowledge is MotionKnowledge.REFERENCE_EPOCH_ONLY

    at_epoch = at_reference_epoch(state)
    assert at_epoch is not None
    assert at_epoch.is_at_reference_epoch
    assert not at_epoch.motion_applied
    assert at_epoch.blockers == ()
    assert at_epoch.is_publishable
    assert at_epoch.position is state.position

    # One day later, the same star cannot answer at all.
    later = propagate_astrometry(state, astropy_time(2451546.0))
    assert not later.is_publishable
    assert PROPER_MOTION_NOT_MEASURED in later.blockers


def test_a_state_without_a_reference_epoch_blocks_cross_epoch_publication():
    """An undated position is not attached to any instant.

    So it cannot be moved to another one, and it cannot be published *at*
    one either - which is the gate the whole slice exists to enforce.
    """
    undated = AstrometricState(
        position=sky_position("undated", 10.0, 20.0, parallax_mas=measured(10.0, u.mas)),
        source_catalog="synthetic",
    )
    assert undated.knowledge is MotionKnowledge.NONE
    assert at_reference_epoch(undated) is None

    result = propagate_astrometry(undated, astropy_time(EPOCH_JD))
    assert result is not None
    assert REFERENCE_EPOCH_NOT_STATED in result.blockers
    assert not result.is_publishable
    assert result.status is Status.UNKNOWN

    reasons = absolute_position_blockers(
        undated.position, _resolved_node(), astrometry=result
    )
    assert REFERENCE_EPOCH_NOT_STATED in reasons


# ==========================================================================
# The propagation itself: Astropy owns it, and we check it against Astropy
# ==========================================================================


def test_a_full_state_matches_astropy_apply_space_motion(gaia_cache):
    """Built independently here, so this is a comparison, not a restatement.

    The point is not that ``apply_space_motion`` is correct - it is - but
    that the state handed to it carries the components it should and no
    others, and that the result is read back into the right fields.
    """
    record = gaia_cache.record_for("1019003226022657920")  # HD 80606
    assert record is not None
    state = astrometric_state_from_gaia(record, name="HD 80606")
    assert state.knowledge is MotionKnowledge.FULL_SPACE_MOTION

    target = astropy_time(EPOCH_JD)
    result = propagate_astrometry(state, target)
    assert result.is_publishable

    reference = SkyCoord(
        ra=record.ra * u.deg,
        dec=record.dec * u.deg,
        distance=(1000.0 / record.parallax) * u.pc,
        pm_ra_cosdec=record.pmra * u.mas / u.yr,
        pm_dec=record.pmdec * u.mas / u.yr,
        radial_velocity=record.radial_velocity * u.km / u.s,
        obstime=GAIA_EPOCH,
        frame="icrs",
    ).apply_space_motion(new_obstime=target)

    assert result.position.ra.value_in(u.deg) == pytest.approx(
        float(reference.ra.to_value(u.deg)), abs=1e-12
    )
    assert result.position.dec.value_in(u.deg) == pytest.approx(
        float(reference.dec.to_value(u.deg)), abs=1e-12
    )
    assert result.position.distance.value_in(u.pc) == pytest.approx(
        float(reference.distance.to_value(u.pc)), rel=1e-12
    )


def test_a_high_proper_motion_star_moves_substantially():
    """10 arcsec/yr over 23 years is 230 arcsec. That is not a rounding term.

    The magnitude is checked against the elapsed time and the published rate
    rather than against a stored number, so this fails if the epoch, the
    rate or the direction is wrong - not only if propagation is skipped.
    """
    state = _high_proper_motion()
    target = astropy_time(EPOCH_JD)
    years = float((target - state.reference_epoch).to_value(u.yr))
    assert years > 20.0

    result = propagate_astrometry(state, target)
    assert result.is_publishable
    assert result.motion_applied
    assert not result.is_at_reference_epoch

    before = state.position.skycoord
    after = result.position.skycoord
    moved = before.separation(after).to_value(u.arcsec)

    expected = years * np.hypot(
        state.pm_ra_cosdec.value_in(u.mas / u.yr),
        state.pm_dec.value_in(u.mas / u.yr),
    ) / 1000.0
    assert moved == pytest.approx(expected, rel=2e-3)
    assert moved > 200.0

    # For comparison: skipping the propagation entirely leaves it at zero.
    assert before.separation(before).to_value(u.arcsec) == pytest.approx(0.0)


def test_forward_and_backward_propagation_round_trips():
    """Space motion is a bijection; the code applying it must be too.

    A round trip cannot catch a sign error - that round-trips perfectly -
    which is why the direction is pinned separately above. What it does
    catch is an epoch that is not being subtracted symmetrically, and an
    accumulating conversion.
    """
    state = _high_proper_motion()
    forward = propagate_astrometry(state, astropy_time(EPOCH_JD))
    assert forward.is_publishable

    returned = forward.as_state()
    assert returned.reference_epoch is forward.obstime

    back = propagate_astrometry(returned, state.reference_epoch)
    assert back.is_publishable

    separation = state.position.skycoord.separation(back.position.skycoord)
    assert separation.to_value(u.mas) == pytest.approx(0.0, abs=1.0)
    assert back.position.distance.value_in(u.pc) == pytest.approx(
        state.position.distance.value_in(u.pc), rel=1e-9
    )


def test_the_propagated_motion_is_the_motion_at_the_new_epoch():
    """A star's apparent proper motion is not a constant of the star.

    The tangential velocity is fixed in space, so the angular rate scales
    with distance, and a receding star's radial velocity grows as the space
    velocity turns further along the line of sight. Re-using the catalogue
    rates for a second propagation drops that perspective term - which is
    what makes the round trip above a test of the propagation rather than
    of the caller's bookkeeping.
    """
    state = _high_proper_motion()
    forward = propagate_astrometry(state, astropy_time(EPOCH_JD))

    assert forward.pm_ra_cosdec is not None
    assert forward.radial_velocity is not None
    assert forward.pm_ra_cosdec.status is Status.DERIVED

    # It moved, and the change is far larger than the published uncertainty.
    assert forward.pm_ra_cosdec.value_in(u.mas / u.yr) != pytest.approx(
        state.pm_ra_cosdec.value_in(u.mas / u.yr), rel=1e-9
    )
    assert forward.radial_velocity.value_in(u.km / u.s) != pytest.approx(
        state.radial_velocity.value_in(u.km / u.s), rel=1e-9
    )

    # A direction-only propagation invents no radial velocity on the way out.
    direction_only = AstrometricState(
        position=state.position,
        reference_epoch=state.reference_epoch,
        pm_ra_cosdec=state.pm_ra_cosdec,
        pm_dec=state.pm_dec,
    )
    result = propagate_astrometry(direction_only, astropy_time(EPOCH_JD))
    assert not result.radial_velocity.is_known
    assert not result.as_state().has_radial_velocity


def test_the_right_ascension_wrap_is_handled():
    """A star just below 360 degrees must not propagate to 360.05.

    Hand-written proper-motion propagation gets this wrong in a way that is
    invisible in a separation and catastrophic in a Cartesian position;
    Astropy's angle wrapping is the reason the arithmetic is delegated.
    """
    state = AstrometricState(
        position=sky_position(
            "wrapper", 359.999, 0.0, parallax_mas=measured(100.0, u.mas)
        ),
        source_catalog="synthetic",
        reference_epoch=J2000,
        pm_ra_cosdec=measured(20000.0, u.mas / u.yr, provenance="synthetic"),
        pm_dec=measured(0.0, u.mas / u.yr, provenance="synthetic"),
        radial_velocity=measured(0.0, u.km / u.s, provenance="synthetic"),
    )

    result = propagate_astrometry(state, astropy_time(EPOCH_JD))
    assert result.is_publishable

    ra = result.position.ra.value_in(u.deg)
    assert 0.0 <= ra < 360.0
    # It really did cross: 20 arcsec/yr for 23 years is ~0.128 degrees, so
    # from 359.999 the wrapped answer is a small positive number.
    assert ra < 1.0

    # And the Cartesian position is continuous across the wrap: the radius
    # is still 10 pc to within the light-time term ERFA carries through the
    # propagation, which is parts in 1e6 for a star moving this fast.
    crossed = result.position.cartesian_pc(Frame.ICRS)
    assert np.all(np.isfinite(crossed))
    assert np.linalg.norm(crossed) == pytest.approx(10.0, rel=1e-4)

    # The failure this guards against is an unwrapped angle: 360.127 degrees
    # is a perfectly finite number whose Cartesian position is fine too, so
    # only the bound above catches it.
    assert result.position.ra.value_in(u.deg) != pytest.approx(360.0 + ra, abs=1e-9)


def test_direction_only_propagation_is_not_called_a_3d_position(gaia_cache):
    """TRAPPIST-1's direction moves; its distance is not carried across.

    This is the tier boundary that matters most, because the tempting
    shortcut - propagate the direction, keep the reference-epoch distance -
    produces a complete-looking 3D position that nobody measured.
    """
    record = gaia_cache.record_for("2635476908753563008")
    state = astrometric_state_from_gaia(record, name="TRAPPIST-1")
    assert state.position.has_distance  # it does have a Gaia parallax

    result = propagate_astrometry(state, astropy_time(EPOCH_JD))
    assert result.motion_applied
    assert not result.is_publishable
    assert RADIAL_VELOCITY_NOT_MEASURED in result.blockers
    assert DIRECTION_ONLY_NOT_A_POSITION in result.blockers

    # The direction did move ...
    assert result.position.ra.value_in(u.deg) != pytest.approx(
        state.position.ra.value_in(u.deg), abs=1e-9
    )
    # ... and the distance is explicitly not available at the new epoch,
    # while the reference-epoch one is still on the source state.
    assert not result.position.has_distance
    assert state.position.has_distance


# ==========================================================================
# DIRECTION_ONLY is a model, and must never become a measurement
# ==========================================================================


def _at_radial_velocity(rv_km_s: float | None) -> AstrometricState:
    """The same nearby, fast star with a stated line-of-sight velocity.

    Everything except the radial velocity is held fixed, so any difference
    downstream is attributable to that one component and nothing else.
    """
    return AstrometricState(
        position=sky_position(
            "rv probe", 100.0, 20.0, parallax_mas=measured(500.0, u.mas)
        ),
        source_catalog="synthetic",
        reference_epoch=J2000,
        pm_ra_cosdec=measured(-8000.0, u.mas / u.yr, provenance="synthetic"),
        pm_dec=measured(6000.0, u.mas / u.yr, provenance="synthetic"),
        radial_velocity=(
            unknown(u.km / u.s, provenance="synthetic")
            if rv_km_s is None
            else measured(rv_km_s, u.km / u.s, provenance="synthetic")
        ),
    )


def test_a_missing_radial_velocity_leaves_the_direction_undetermined():
    """The reason DIRECTION_ONLY is a model and not a measurement.

    Astropy's ``apply_space_motion`` assumes RV = 0 when none is supplied,
    and for a finite-distance star the line-of-sight velocity changes the
    *apparent angular* motion through perspective acceleration. So two
    physically possible radial velocities that the catalogue cannot
    distinguish give measurably different sky directions at the same date.

    That difference is the error bar the zero-RV realization does not carry,
    and it is why this tier may not publish a cross-epoch position.
    """
    far = astropy_time(2451545.0 + 365.25 * 400.0)  # four centuries on

    approaching = propagate_astrometry(_at_radial_velocity(-100.0), far)
    receding = propagate_astrometry(_at_radial_velocity(+100.0), far)
    assert approaching.is_publishable and receding.is_publishable

    spread = approaching.position.skycoord.separation(receding.position.skycoord)
    assert spread.to_value(u.arcmin) > 1.0, spread.to_value(u.arcmin)

    # The zero-RV realization sits between them and is not either of them.
    unknown_rv = propagate_astrometry(_at_radial_velocity(None), far)
    assert unknown_rv.knowledge is MotionKnowledge.DIRECTION_ONLY
    assert not unknown_rv.is_publishable

    for named in (approaching, receding):
        offset = unknown_rv.position.skycoord.separation(named.position.skycoord)
        assert offset.to_value(u.arcsec) > 1.0

    # And the result says so, in the blockers and on the values themselves.
    assert DIRECTION_ONLY_IS_MODEL_DEPENDENT in unknown_rv.blockers
    assert ZERO_RV_APPROXIMATION in unknown_rv.position.ra.note
    assert "zero-RV" in unknown_rv.position.ra.note


def test_direction_only_never_returns_a_known_radial_velocity():
    """Astropy hands back a differential; this must not read one off it.

    A unit-spherical propagation still produces a velocity object, and
    lifting its radial component would turn "nobody measured this" into a
    measured number in one assignment.
    """
    result = propagate_astrometry(_at_radial_velocity(None), astropy_time(EPOCH_JD))

    assert result.radial_velocity is not None
    assert not result.radial_velocity.is_known
    assert result.radial_velocity.value is None
    assert result.radial_velocity.status is Status.UNKNOWN

    # The proper motions it *did* produce are labelled with the right model.
    assert result.pm_ra_cosdec.is_known
    assert result.pm_ra_cosdec.note == ZERO_RV_APPROXIMATION
    assert result.pm_ra_cosdec.note != SPACE_MOTION_MODEL


def test_direction_only_as_state_preserves_an_unknown_radial_velocity():
    """The chaining step, where a model realization could become evidence."""
    result = propagate_astrometry(_at_radial_velocity(None), astropy_time(EPOCH_JD))
    chained = result.as_state()

    assert not chained.has_radial_velocity
    assert chained.radial_velocity.status is Status.UNKNOWN
    assert chained.knowledge is MotionKnowledge.DIRECTION_ONLY
    assert chained.knowledge is not MotionKnowledge.FULL_SPACE_MOTION


def test_chained_direction_only_cannot_promote_itself_to_full_space_motion():
    """Repeat the hop; the tier must not creep upward.

    This is the failure the audit named: unknown RV -> zero-RV realization
    -> resulting differential -> a new state with an apparently known RV.
    Ten hops is more than any caller would do and cheap to check.
    """
    state = _at_radial_velocity(None)
    for step in range(10):
        propagated = propagate_astrometry(
            state, astropy_time(EPOCH_JD + 400.0 * step)
        )
        assert propagated.knowledge is MotionKnowledge.DIRECTION_ONLY, step
        assert not propagated.is_publishable, step
        assert RADIAL_VELOCITY_NOT_MEASURED in propagated.blockers, step
        state = propagated.as_state()

    assert not state.has_radial_velocity
    assert state.knowledge is MotionKnowledge.DIRECTION_ONLY


def test_chaining_rebases_onto_the_observation_rather_than_the_realization():
    """A realization is never propagated a second time.

    Applying the zero-RV approximation twice would look exactly like one
    careful propagation. So a state that came out of a propagation records
    where it came from, and the next propagation starts there.
    """
    original = _at_radial_velocity(None)
    midpoint = propagate_astrometry(original, astropy_time(EPOCH_JD)).as_state()

    assert midpoint.is_realization
    assert midpoint.observed_root() is original
    assert not original.is_realization
    assert original.observed_root() is original

    far = astropy_time(EPOCH_JD + 40000.0)
    via_midpoint = propagate_astrometry(midpoint, far)
    direct = propagate_astrometry(original, far)

    # Identical, because both propagated the observation exactly once.
    assert via_midpoint.position.ra.value_in(u.deg) == pytest.approx(
        direct.position.ra.value_in(u.deg), abs=1e-12
    )
    assert via_midpoint.position.dec.value_in(u.deg) == pytest.approx(
        direct.position.dec.value_in(u.deg), abs=1e-12
    )
    assert any("Rebased" in line for line in via_midpoint.describe())


def test_full_space_motion_may_chain_and_stays_full_space_motion():
    """The tier that has a measured RV keeps it, and keeps its rank."""
    state = _at_radial_velocity(-100.0)
    assert state.knowledge is MotionKnowledge.FULL_SPACE_MOTION

    for step in range(5):
        propagated = propagate_astrometry(
            state, astropy_time(EPOCH_JD + 400.0 * step)
        )
        assert propagated.knowledge is MotionKnowledge.FULL_SPACE_MOTION, step
        assert propagated.is_publishable, step
        assert propagated.radial_velocity.is_known, step
        assert propagated.position.ra.note == SPACE_MOTION_MODEL, step
        state = propagated.as_state()

    assert state.has_radial_velocity
    assert state.knowledge is MotionKnowledge.FULL_SPACE_MOTION


def test_a_direction_only_state_cannot_unlock_a_planet_position():
    """The tier boundary, enforced where it matters."""
    state = _at_radial_velocity(None)
    propagated = propagate_astrometry(state, astropy_time(EPOCH_JD))

    row = absolute_planet_position(
        state.position,
        _timed(np.array([0.3, 0.0, 0.0]), obstime=propagated.obstime),
        node=_resolved_node(),
        astrometry=propagated,
    )
    assert not row.is_known
    assert RADIAL_VELOCITY_NOT_MEASURED in row.note
    assert DIRECTION_ONLY_IS_MODEL_DEPENDENT in row.note

    # The same star with a measured RV publishes.
    with_rv = _at_radial_velocity(-100.0)
    ok = absolute_planet_position(
        with_rv.position,
        _timed(np.array([0.3, 0.0, 0.0]), obstime=astropy_time(EPOCH_JD)),
        node=_resolved_node(),
        astrometry=propagate_astrometry(with_rv, astropy_time(EPOCH_JD)),
    )
    assert ok.is_known


# ==========================================================================
# The absolute-orientation gate: the node is necessary and not sufficient
# ==========================================================================


def _eccentric(**kwargs):
    """A deliberately non-circular orbit, so periapsis is a real direction."""
    kwargs.setdefault("eccentricity", 0.4)
    return _oriented_elements(**kwargs)


def test_a_normalised_inclination_never_publishes_an_absolute_position():
    """An orbit drawn face-on has no plane in space.

    This is the hole the second audit found. Before it, a planet whose
    inclination was normalised for display reached a published ICRS
    coordinate on the strength of a tagged node alone - and the number
    looked entirely ordinary.
    """
    host = _measured_zero_motion()
    propagated = propagate_astrometry(host, astropy_time(EPOCH_JD))

    unresolved = _eccentric(inclination_deg=None)
    assert INCLINATION_UNRESOLVED in absolute_orientation_blockers(unresolved)

    row = absolute_planet_position(
        host.position,
        _timed(np.array([0.3, 0.0, 0.0]), elements=unresolved),
        astrometry=propagated,
    )
    assert not row.is_known
    assert INCLINATION_UNRESOLVED in row.note

    # The display normalisation must not launder it either: for_display
    # replaces the missing angle with an ASSUMED zero, and an assumption is
    # not an observation.
    normalised = unresolved.for_display()
    assert normalised.inclination.is_known
    assert normalised.inclination.status is Status.ASSUMED_FOR_VISUALIZATION
    assert INCLINATION_UNRESOLVED in absolute_orientation_blockers(normalised)


def test_a_transit_epoch_with_an_assumed_omega_does_not_publish():
    """Concrete failure mode A: an observed instant is not an orientation.

    At transit the argument of latitude is pi/2, so nu_transit = pi/2 -
    omega. For an eccentric orbit the mapping nu -> E -> M is nonlinear in
    omega, so advancing M(t) = M_transit + n(t - t_transit) does not give a
    unique physical position without the real omega.

    The *timing* is a genuine observation, and the time gate correctly
    accepts it. The orientation gate is a separate question and refuses.
    """
    host = _measured_zero_motion()
    propagated = propagate_astrometry(host, astropy_time(EPOCH_JD))

    transit = _timed(
        np.array([0.3, 0.0, 0.0]),
        phase=PhaseSolution(0.4, PhaseProvenance.TRANSIT_CONJUNCTION_NORMALIZED),
        elements=_eccentric(omega_deg=41.0, convention=PeriastronConvention.AS_REPORTED),
    )
    # The time gate is satisfied ...
    assert transit.is_time_constrained
    # ... and the orientation gate is not.
    assert not transit.is_orientation_resolved

    row = absolute_planet_position(host.position, transit, astrometry=propagated)
    assert not row.is_known
    assert PERIASTRON_CONVENTION_UNSTATED in row.note
    assert PLANET_PHASE_NOT_CONSTRAINED not in row.note


def test_a_periastron_epoch_with_an_unknown_omega_does_not_publish():
    """Concrete failure mode B: M = 0 exactly, and periapsis points nowhere.

    A periastron epoch gives the mean anomaly directly and needs no omega at
    all, which is why the phase model calls it CONSTRAINED. An absolute 3D
    position still has to know which direction periapsis points within the
    orbital plane, and that is a different measurement.
    """
    host = _measured_zero_motion()
    propagated = propagate_astrometry(host, astropy_time(EPOCH_JD))

    periastron = _timed(
        np.array([0.3, 0.0, 0.0]),
        phase=PhaseSolution(0.0, PhaseProvenance.PERIASTRON_EPOCH),
        elements=_eccentric(omega_deg=None, convention=PeriastronConvention.UNKNOWN),
    )
    assert periastron.phase.status.is_observationally_anchored
    assert periastron.is_time_constrained

    row = absolute_planet_position(host.position, periastron, astrometry=propagated)
    assert not row.is_known
    assert PERIAPSIS_DIRECTION_UNRESOLVED in row.note


def test_an_unstated_periastron_convention_blocks():
    """A real number under a convention nobody recorded is 180 degrees wide.

    RV papers habitually report the star's reflex orbit and transit papers
    the planet's. For an eccentric orbit the difference puts the planet on
    the wrong side of its star - and the archive carries no machine-readable
    column saying which it is, so AS_REPORTED is the honest default and it
    blocks.
    """
    unstated = _eccentric(convention=PeriastronConvention.AS_REPORTED)
    reasons = absolute_orientation_blockers(unstated)
    assert PERIASTRON_CONVENTION_UNSTATED in reasons
    # One root cause, one sentence: the convention message already explains
    # why the direction is unresolved.
    assert PERIAPSIS_DIRECTION_UNRESOLVED not in reasons

    # The resolved planet-frame value is what carries the verdict.
    assert not unstated.argument_of_periapsis_planet.is_scientific
    assert (
        unstated.argument_of_periapsis_planet.status
        is Status.ASSUMED_FOR_VISUALIZATION
    )


def test_a_stellar_reflex_omega_converted_to_the_planet_frame_passes():
    """The gate is a gate, not a wall.

    A publication that states it reported the star's reflex orbit has said
    something real; +180 degrees is a documented transform of a stated
    convention, so the result is DERIVED and publishable - not a guess.
    """
    reflex = _eccentric(
        omega_deg=41.0, convention=PeriastronConvention.STELLAR_REFLEX
    )
    resolved = reflex.argument_of_periapsis_planet
    assert resolved.status is Status.DERIVED
    assert resolved.is_scientific
    assert resolved.value_in(u.deg) == pytest.approx(41.0 + 180.0, abs=1e-9)

    assert absolute_orientation_blockers(reflex) == ()

    host = _measured_zero_motion()
    row = absolute_planet_position(
        host.position,
        _timed(np.array([0.3, 0.0, 0.0]), elements=reflex),
        astrometry=propagate_astrometry(host, astropy_time(EPOCH_JD)),
    )
    assert row.is_known
    assert row.note == ""


def test_a_display_normalised_orientation_never_passes_as_physical():
    """for_display fills angles so a scene can be drawn. That is all it does.

    Every angle it substitutes comes back ASSUMED_FOR_VISUALIZATION, and the
    gate reads status rather than presence - so running an element set
    through the renderer's normalisation cannot make it publishable.
    """
    bare = _eccentric(inclination_deg=None, omega_deg=None,
                      convention=PeriastronConvention.UNKNOWN)
    shown = bare.for_display()

    # Every angle now has a value ...
    assert shown.inclination.is_known
    assert shown.argument_of_periastron.is_known
    # ... and none of them is an observation.
    assert not shown.inclination.is_scientific
    assert not shown.argument_of_periastron.is_scientific

    reasons = absolute_orientation_blockers(shown)
    assert INCLINATION_UNRESOLVED in reasons
    assert PERIAPSIS_DIRECTION_UNRESOLVED in reasons
    assert absolute_orientation_blockers(bare) == reasons


def test_a_fully_observed_orientation_passes_the_gate():
    """Measured i, planet-frame omega and a resolved node: nothing left."""
    resolved = _eccentric()
    assert absolute_orientation_blockers(resolved) == ()
    assert resolved.inclination.is_scientific
    assert resolved.argument_of_periapsis_planet.is_scientific
    assert node_sense_of(resolved.longitude_of_ascending_node) is NodeSense.RESOLVED


def test_the_node_gates_are_necessary_and_not_sufficient():
    """The audit's finding, stated as one assertion.

    A perfect node with a normalised plane is still no orientation, and a
    perfect plane with an untagged node is still no orientation.
    """
    plane_only = _eccentric(node=measured(0.7, u.rad, provenance="test"))
    assert INCLINATION_UNRESOLVED not in absolute_orientation_blockers(plane_only)
    assert NODE_SENSE_UNRESOLVED in absolute_orientation_blockers(plane_only)

    node_only = _eccentric(inclination_deg=None, omega_deg=None,
                           convention=PeriastronConvention.UNKNOWN)
    reasons = absolute_orientation_blockers(node_only)
    assert NODE_SENSE_UNRESOLVED not in reasons
    assert NODE_CONVENTION_UNSTATED not in reasons
    assert INCLINATION_UNRESOLVED in reasons


def test_every_failing_orientation_gate_is_reported_together():
    """A row blocked for four reasons should say four."""
    worst = _oriented_elements(
        inclination_deg=None,
        omega_deg=None,
        convention=PeriastronConvention.UNKNOWN,
        node=measured(0.7, u.rad, provenance="test"),  # value, no metadata
    )
    reasons = absolute_orientation_blockers(worst)
    assert INCLINATION_UNRESOLVED in reasons
    assert PERIAPSIS_DIRECTION_UNRESOLVED in reasons
    assert NODE_CONVENTION_UNSTATED in reasons
    assert NODE_SENSE_UNRESOLVED in reasons
    assert len(reasons) == 4


def test_timing_and_orientation_blockers_are_reported_side_by_side():
    """Two epistemic dimensions, both named, neither standing in for the other."""
    host = _measured_zero_motion()
    propagated = propagate_astrometry(host, astropy_time(EPOCH_JD))

    row = absolute_planet_position(
        host.position,
        _timed(
            np.array([0.3, 0.0, 0.0]),
            obstime=propagated.obstime,
            phase=PhaseSolution(1.2, PhaseProvenance.ASSUMED_ZERO_PHASE),
            elements=_eccentric(inclination_deg=None),
        ),
        astrometry=propagated,
    )
    assert not row.is_known
    assert PLANET_PHASE_NOT_CONSTRAINED in row.note
    assert INCLINATION_UNRESOLVED in row.note


def test_the_planet_to_star_distance_inherits_the_orientation_gate():
    """Inherited, not restated: a second copy could drift from the first."""
    host = _measured_zero_motion()
    target = _high_proper_motion()
    t = astropy_time(EPOCH_JD)
    offset = np.array([0.3, -0.1, 0.05])

    def distance(elements):
        return planet_to_star_distance(
            host.position,
            _timed(offset, obstime=t, elements=elements),
            target.position,
            astrometry=propagate_astrometry(host, t),
            other_astrometry=propagate_astrometry(target, t),
        )

    blocked = distance(_eccentric(inclination_deg=None))
    assert not blocked.is_known
    assert INCLINATION_UNRESOLVED in blocked.note

    assert distance(_eccentric()).is_known


def test_a_circular_orbit_is_conservatively_still_blocked_on_periapsis():
    """A deliberate refusal, not an oversight - and the reasoning is recorded.

    For e = 0 there is no periapsis, so omega is not a physical degree of
    freedom and demanding it looks like an over-refusal. Lifting it correctly
    needs the *phase anchor* as well:

    * with a transit or conjunction anchor, nu_t = pi/2 - omega at the
      anchor, so the argument of latitude is u(t) = pi/2 + n(t - t_t) and
      omega cancels exactly. Such an orbit genuinely does not need it;
    * with a periastron anchor, or a mean anomaly quoted at an epoch, the
      anchor is measured *from* periapsis - which does not exist at e = 0 -
      so the epoch has no meaning and nothing is recovered.

    Phase and orientation are separate dimensions here on purpose, and no
    catalogue row in the snapshot can currently reach this case: every real
    exoplanet is already blocked on the node. So the conservative answer
    ships and the analysis stays written down.
    """
    circular = _oriented_elements(
        eccentricity=0.0, omega_deg=None, convention=PeriastronConvention.UNKNOWN
    )
    assert circular.eccentricity.is_scientific
    assert circular.eccentricity.value == 0.0

    reasons = absolute_orientation_blockers(circular)
    assert PERIAPSIS_DIRECTION_UNRESOLVED in reasons

    # The refusal is documented where someone would go to lift it.
    assert "transit anchor" in CIRCULAR_ORBIT_PERIAPSIS_NOTE
    assert "periastron" in CIRCULAR_ORBIT_PERIAPSIS_NOTE


def test_no_real_catalogue_planet_publishes_an_absolute_position(
    hd80606, hd219134, trappist1
):
    """The whole snapshot, through the misuse-proof entry point.

    Every gate is now real rather than deferred, and every planet is still
    withheld - on the node, which no exoplanet in the catalogue has. That is
    a statement about the catalogue, not a placeholder.
    """
    for system in (hd80606, hd219134, trappist1):
        for record in system.planets:
            row = system.absolute_planet_position(record, EPOCH_JD)
            assert not row.is_known, (system.star.name, record.name)
            assert NODE_SENSE_UNRESOLVED in row.note, record.name

            # The astrometric gate is genuinely satisfied where the science
            # supports it, so the refusal is never resting on it.
            if system.astrometry.knowledge is MotionKnowledge.FULL_SPACE_MOTION:
                assert ASTROMETRY_NOT_PROPAGATED not in row.note, record.name

            # And the orientation gate is doing real work alongside the node:
            # HD 80606 b has a published argument of periastron under an
            # unstated convention, and Kepler-11's planets have none at all.
            # Before the second audit neither was checked.
            assert (
                PERIASTRON_CONVENTION_UNSTATED in row.note
                or PERIAPSIS_DIRECTION_UNRESOLVED in row.note
                or INCLINATION_UNRESOLVED in row.note
            ), (record.name, row.note)

    hd = hd80606.planet("HD 80606 b")
    assert PERIASTRON_CONVENTION_UNSTATED in hd80606.absolute_planet_position(
        hd, EPOCH_JD
    ).note


# ==========================================================================
# The common-time rule
# ==========================================================================


def test_the_host_tangent_basis_uses_the_propagated_ra_and_dec():
    """The basis and the origin must come from the same position.

    Building the triad from the catalogue RA/Dec while adding the offset to
    the propagated origin rotates the offset away from the position it
    belongs to. For a fast star that rotation is arcminutes, which at parsec
    range dwarfs the AU-scale offset entirely - and the result still looks
    like a perfectly ordinary coordinate.
    """
    state = _high_proper_motion()
    target = astropy_time(EPOCH_JD)
    propagated = propagate_astrometry(state, target)
    assert propagated.is_publishable

    offset_au = np.array([1.0, 0.0, 0.0])  # one AU due East in the sky frame
    row = absolute_planet_position(
        state.position,
        _timed(offset_au, obstime=target),
        node=_resolved_node(),
        astrometry=propagated,
    )
    assert row.is_known

    one_au_pc = float((1.0 * u.au).to_value(u.pc))
    origin = propagated.position.cartesian_pc(Frame.ICRS)
    offset_pc = np.asarray(row.values) - origin

    moved_basis = tangent_basis(
        propagated.position.ra.value_in(u.deg),
        propagated.position.dec.value_in(u.deg),
    )
    catalogue_basis = tangent_basis(
        state.position.ra.value_in(u.deg), state.position.dec.value_in(u.deg)
    )

    assert np.allclose(offset_pc, moved_basis.east * one_au_pc, atol=1e-18)
    # The two bases really are different, so this is a discriminating test.
    assert not np.allclose(moved_basis.east, catalogue_basis.east, atol=1e-9)
    assert not np.allclose(offset_pc, catalogue_basis.east * one_au_pc, atol=1e-18)


def test_the_origin_is_the_propagated_position_not_the_catalogue_one():
    """The whole row moves with the star, not just the offset's basis."""
    state = _high_proper_motion()
    propagated = propagate_astrometry(state, astropy_time(EPOCH_JD))

    row = absolute_planet_position(
        state.position,
        _timed(np.zeros(3), obstime=propagated.obstime),
        node=_resolved_node(),
        astrometry=propagated,
    )
    assert row.is_known
    assert np.allclose(
        row.values, propagated.position.cartesian_pc(Frame.ICRS), atol=1e-15
    )
    assert not np.allclose(
        row.values, state.position.cartesian_pc(Frame.ICRS), atol=1e-9
    )


def test_a_planet_to_star_distance_needs_all_three_clocks_to_agree():
    """Three clocks, not two: host, target star **and** planet.

    Two of the three were checkable from the start. The planet's was not,
    because a bare offset array has no instant - so a host propagated to
    2035 and an orbit propagated to 2025 added together perfectly and
    produced a coordinate about a different night.

    All three permutations of one-out-of-step are refused, because a check
    that only looked at the pair it happened to be handed would pass two of
    them.
    """
    host = _measured_zero_motion()
    target = _high_proper_motion()
    offset_au = np.array([0.3, -0.1, 0.05])

    t1 = astropy_time(EPOCH_JD)
    t2 = astropy_time(EPOCH_JD + 3652.5)  # ten years later

    def distance(host_time, target_time, planet_time):
        return planet_to_star_distance(
            host.position,
            _timed(offset_au, obstime=planet_time),
            target.position,
            node=_resolved_node(),
            astrometry=propagate_astrometry(host, host_time),
            other_astrometry=propagate_astrometry(target, target_time),
        )

    agreed = distance(t1, t1, t1)
    assert agreed.is_known

    for label, times in (
        ("planet out of step", (t1, t1, t2)),
        ("target out of step", (t1, t2, t1)),
        ("host out of step", (t2, t1, t1)),
    ):
        result = distance(*times)
        assert not result.is_known, label
        assert result.value is None, label
        assert (
            COMMON_TIME_VIOLATED in result.note
            or PLANET_TIME_MISMATCH in result.note
        ), (label, result.note)

    # Moving all three together is a different answer, not an error - and
    # the size of that difference is what a mismatch was hiding.
    together = distance(t2, t2, t2)
    assert together.is_known
    assert together.value_in(u.pc) != pytest.approx(agreed.value_in(u.pc), rel=1e-6)


def test_an_undated_planet_offset_cannot_open_the_publication_gate():
    """A bare array has no instant, so it cannot be shown to match one.

    This is the structural half of the common-time rule. The check is not
    "did the caller remember to use the same time_jd" - it is that the type
    which could be wrong is not accepted at all.
    """
    host = _measured_zero_motion()
    propagated = propagate_astrometry(host, astropy_time(EPOCH_JD))
    offset_au = np.array([0.3, -0.1, 0.05])

    bare = absolute_planet_position(
        host.position, offset_au, node=_resolved_node(), astrometry=propagated
    )
    assert not bare.is_known
    assert PLANET_STATE_NOT_TIMED in bare.note

    dated = absolute_planet_position(
        host.position,
        _timed(offset_au, obstime=propagated.obstime),
        node=_resolved_node(),
        astrometry=propagated,
    )
    assert dated.is_known
    assert dated.note == ""

    # Only datedness differs: the vector is the one the blocked call would
    # have produced, which is exactly why the type had to carry the instant.
    expected = np.asarray(
        propagated.position.cartesian_pc(Frame.ICRS)
    ) + system_offset_to_icrs_pc(propagated.position, offset_au)
    assert np.allclose(dated.values, expected, atol=1e-18)


def test_an_assumed_phase_is_dated_and_still_not_publishable():
    """Tracking the epoch of an arbitrary zero does not make it a date.

    A planet with no published epoch is advanced at the correct rate from
    the Unix epoch. Its obstime is exact and matches the host's exactly, so
    every clock check passes - and the position is still a picture of the
    motion rather than a claim about that night.
    """
    host = _measured_zero_motion()
    propagated = propagate_astrometry(host, astropy_time(EPOCH_JD))

    assumed_phase = _timed(
        np.array([0.3, -0.1, 0.05]),
        obstime=propagated.obstime,
        phase=PhaseSolution(1.2, PhaseProvenance.ASSUMED_ZERO_PHASE),
    )
    assert assumed_phase.at_same_time_as(propagated)
    assert not assumed_phase.is_time_constrained

    row = absolute_planet_position(
        host.position, assumed_phase, node=_resolved_node(), astrometry=propagated
    )
    assert not row.is_known
    assert PLANET_PHASE_NOT_CONSTRAINED in row.note


def test_a_transit_normalised_phase_is_still_an_observed_instant():
    """The time gate must not double-count the orientation gates.

    A transit epoch read through a normalised argument of periastron is
    PARTIALLY_CONSTRAINED: the *timing* is a real observation and only the
    in-plane orientation was normalised. That orientation is already gated
    twice, by node convention and node sense, so refusing it here as well
    would withhold a published epoch for a reason that has nothing to do
    with when the planet was.
    """
    partial = _timed(
        np.array([0.3, 0.0, 0.0]),
        phase=PhaseSolution(0.4, PhaseProvenance.TRANSIT_CONJUNCTION_NORMALIZED),
    )
    assert partial.phase.is_assumed  # not fully constrained ...
    assert partial.is_time_constrained  # ... but the instant is observed

    host = _measured_zero_motion()
    row = absolute_planet_position(
        host.position,
        partial,
        node=_resolved_node(),
        astrometry=propagate_astrometry(host, astropy_time(EPOCH_JD)),
    )
    assert row.is_known
    assert PLANET_PHASE_NOT_CONSTRAINED not in row.note


def test_instants_are_compared_not_julian_day_numbers():
    """The same instant in two scales has two Julian dates.

    TDB and UTC differ by about 69 s today, and TCB adds ~20 s more. A
    comparison on the numbers would refuse a matched pair reached through
    different scales, and accept a genuinely mismatched pair that happened
    to share a number.
    """
    instant = astropy_time(EPOCH_JD, TimeScale.BJD_TDB)
    restated = instant.utc

    assert float(restated.jd) != pytest.approx(float(instant.jd), abs=1e-9)
    assert same_instant(instant, restated)

    host = _measured_zero_motion()
    propagated = propagate_astrometry(host, instant)
    row = absolute_planet_position(
        host.position,
        _timed(np.array([0.2, 0.0, 0.0]), obstime=restated),
        node=_resolved_node(),
        astrometry=propagated,
    )
    assert row.is_known, row.note


def test_the_orbital_clock_has_one_conversion_point():
    """astropy_time and orbital_time_jd are exact inverses.

    So the orbital propagator is handed the instant expressed on the axis
    its own published epoch sits on. The failure this prevents is call sites
    independently reaching for time.jd, time.tdb.jd, time.tcb.jd or
    time.utc.jd - which differ by up to about 89 s, invisible in a rendered
    orbit and fatal in a transit ephemeris.
    """
    for scale in (
        TimeScale.BJD_TDB,
        TimeScale.JD_UTC,
        TimeScale.HJD_UTC,
        TimeScale.BKJD,
        TimeScale.JD_UNSPECIFIED,
    ):
        instant = astropy_time(EPOCH_JD, scale)
        assert orbital_time_jd(instant, scale) == pytest.approx(EPOCH_JD, abs=1e-9)

    # Reading one instant through the wrong scale is the error being
    # prevented, and it is a real number of seconds, not a rounding term.
    tdb = astropy_time(EPOCH_JD, TimeScale.BJD_TDB)
    drift_days = orbital_time_jd(tdb, TimeScale.JD_UTC) - EPOCH_JD
    assert abs(drift_days * 86400.0) > 60.0

    # A precise target time does not make an unstated published scale known.
    assert not TimeScale.JD_UNSPECIFIED.is_determinate
    assert TimeScale.JD_UNSPECIFIED.uncertainty_seconds_at(EPOCH_JD) > 500.0


def test_the_slice_propagates_the_host_to_the_time_it_propagates_the_orbit(hd80606):
    """One clock, one conversion point, one instant.

    The slice owns the Julian date, so it owns the pairing: there is no
    argument a caller can get wrong, because there is no argument.
    """
    record = hd80606.planet("HD 80606 b")
    for time_jd in (EPOCH_JD, EPOCH_JD + 1000.0):
        propagated = hd80606.host_astrometry(time_jd)
        assert propagated is not None
        assert propagated.obstime_jd == pytest.approx(
            float(hd80606.obstime(time_jd).jd), abs=1e-12
        )
        assert hd80606.state(record, time_jd) is not None

    early = hd80606.host_astrometry(EPOCH_JD)
    late = hd80606.host_astrometry(EPOCH_JD + 1000.0)
    assert not early.at_same_time_as(late)
    assert early.at_same_time_as(hd80606.host_astrometry(EPOCH_JD))


def test_a_target_star_without_a_propagated_state_blocks_the_distance():
    """Both ends of a separation need an epoch, not just the near one."""
    host = _measured_zero_motion()
    target = _high_proper_motion()

    result = planet_to_star_distance(
        host.position,
        _timed(np.array([0.3, -0.1, 0.05])),
        target.position,
        node=_resolved_node(),
        astrometry=propagate_astrometry(host, astropy_time(EPOCH_JD)),
        other_astrometry=None,
    )
    assert not result.is_known
    assert TARGET_ASTROMETRY_NOT_PROPAGATED in result.note


def test_the_astrometric_api_takes_a_time_not_a_float():
    """A float has no scale, so three consumers could read it three ways.

    ``astropy_time`` is the one conversion point from the explorer's clock
    to the time stellar propagation runs on, and everything downstream takes
    the :class:`~astropy.time.Time` it produces.
    """
    import inspect

    signature = inspect.signature(propagate_astrometry)
    assert "obstime" in signature.parameters
    assert "time_jd" not in signature.parameters

    stated = astropy_time(EPOCH_JD, TimeScale.BJD_TDB)
    assert isinstance(stated, Time)
    assert stated.scale == "tdb"
    assert astropy_time(EPOCH_JD, TimeScale.JD_UTC).scale == "utc"
    # An unstated scale is assumed explicitly, not left to a caller.
    assert astropy_time(EPOCH_JD, TimeScale.JD_UNSPECIFIED).scale == "tdb"

    with pytest.raises(ValueError):
        astropy_time(float("nan"))


# ==========================================================================
# The escape hatch is gone
# ==========================================================================


def test_epoch_resolved_is_no_longer_a_production_argument():
    """The boolean is removed from the source, not merely defaulted to False.

    A parameter that unlocks every scientific gate when set to ``True`` is
    one edit away from being set to ``True``. Deleting it means the gate can
    only be opened by supplying a state that had to be propagated.
    """
    # Comments and docstrings are stripped first: these modules *document*
    # the argument they removed, and scanning raw text would flag the
    # explanation as the offence. This is the same ``_code_only`` trick the
    # architecture tests use for the constructs they forbid.
    offenders = [
        path.relative_to(SRC)
        for path in SRC.rglob("*.py")
        if "epoch_resolved" in _code_only(path)
    ]
    assert offenders == []

    import inspect

    for function in (absolute_planet_position, planet_to_star_distance, absolute_position_blockers):
        assert "epoch_resolved" not in inspect.signature(function).parameters


def test_omitting_the_astrometry_blocks_rather_than_defaults_open():
    """The gate's default is closed, and closing it needs no argument."""
    reasons = absolute_position_blockers(
        _measured_zero_motion().position, _resolved_node()
    )
    assert ASTROMETRY_NOT_PROPAGATED in reasons

    row = absolute_planet_position(
        _measured_zero_motion().position,
        np.array([0.2, 0.0, 0.0]),
        node=_resolved_node(),
    )
    assert not row.is_known
    assert ASTROMETRY_NOT_PROPAGATED in row.note


def test_the_planet_to_star_distance_never_falls_back_to_host_to_star(
    hd80606, hd219134
):
    """A blocked planet distance is not silently answered with the star's.

    The fallback would be plausible - the two differ by an AU at parsec
    range - and would answer a question nobody asked, with nothing on the
    number saying which question it answered.
    """
    record = hd80606.planet("HD 80606 b")
    state = hd80606.state(record, EPOCH_JD)

    result = planet_to_star_distance(
        hd80606.star.position,
        state.position,
        hd219134.star.position,
        node=record.elements.longitude_of_ascending_node,
        astrometry=hd80606.host_astrometry(EPOCH_JD),
        other_astrometry=hd219134.host_astrometry(EPOCH_JD),
    )

    # The node is unmeasured for every real exoplanet, so this is blocked -
    # by the node, with the astrometry now genuinely resolved.
    assert not result.is_known
    assert result.value is None
    assert NODE_SENSE_UNRESOLVED in result.note
    assert ASTROMETRY_NOT_PROPAGATED not in result.note

    star_to_star = np.linalg.norm(
        hd219134.star.position.cartesian_pc(Frame.ICRS)
        - hd80606.star.position.cartesian_pc(Frame.ICRS)
    )
    assert result.value != star_to_star


def test_the_node_gates_stay_independent_of_the_astrometric_one():
    """Three gates, three reasons, no coupling.

    Resolving the epoch must not relax the node, and an unstated node
    convention must not be excused by an excellent proper motion. Each
    combination is checked because the failure being guarded against is one
    gate quietly standing in for another.
    """
    host = _measured_zero_motion()
    propagated = propagate_astrometry(host, astropy_time(EPOCH_JD))

    bare = measured(0.7, u.rad, provenance="test")
    sense_only = replace(
        bare,
        extra={
            "node_sense": NodeSense.RESOLVED.value,
            "node_sense_evidence": NodeSenseEvidence.ORBITAL_RV_SOLUTION.value,
        },
    )
    convention_only = replace(
        bare, extra={"node_convention": NodeConvention.PA_EAST_OF_NORTH_RECEDING.value}
    )

    # Astrometry resolved, node convention missing.
    assert NODE_CONVENTION_UNSTATED in absolute_position_blockers(
        host.position, sense_only, astrometry=propagated
    )
    # Astrometry resolved, node sense unresolved.
    assert NODE_SENSE_UNRESOLVED in absolute_position_blockers(
        host.position, convention_only, astrometry=propagated
    )
    # Node fully resolved, astrometry missing.
    assert absolute_position_blockers(host.position, _resolved_node()) == [
        ASTROMETRY_NOT_PROPAGATED
    ]
    # Both resolved: nothing left.
    assert (
        absolute_position_blockers(
            host.position, _resolved_node(), astrometry=propagated
        )
        == []
    )


def test_a_systemic_radial_velocity_still_does_not_resolve_a_node():
    """C3.6 gives stellar RV a real job. It is not this one.

    Systemic radial velocity is exactly what full 3D space motion needs, and
    it remains exactly what a planet's ascending node cannot be resolved by:
    it describes the whole system's motion relative to the Sun and says
    nothing about which of two nodes recedes. The type boundary is what
    keeps the new use from leaking into the old refusal.
    """
    host = _measured_zero_motion()
    assert host.has_radial_velocity  # useful for space motion ...

    node = replace(
        measured(0.7, u.rad, provenance="test"),
        extra={
            "node_convention": NodeConvention.PA_EAST_OF_NORTH_RECEDING.value,
            "node_sense": NodeSense.RESOLVED.value,
            "node_sense_evidence": NodeSenseEvidence.SYSTEMIC_RADIAL_VELOCITY.value,
        },
    )
    # ... and still not evidence for the node.
    assert not NodeSenseEvidence.SYSTEMIC_RADIAL_VELOCITY.resolves_node
    reasons = absolute_position_blockers(
        host.position,
        node,
        astrometry=propagate_astrometry(host, astropy_time(EPOCH_JD)),
    )
    assert NODE_SENSE_UNRESOLVED in reasons


# ==========================================================================
# The UNSPECIFIED node-convention hardening
# ==========================================================================


def test_a_measured_node_without_a_stated_convention_is_not_drawn_at_its_value():
    """An angle whose convention nobody recorded is not a position angle.

    Feeding it to the rotation would silently assert the standard
    convention - the failure mode that is right most of the time and
    therefore the worst one to have. The documented normalisation is drawn
    instead, and the reason is available rather than implied.

    Nothing in the NASA archive can trigger this today, because it publishes
    no node at all. It is written now because C3.6 is the slice where a
    provider starts ingesting real astrometric values.
    """
    unstated = measured(0.7, u.rad, provenance="test")
    resolution = resolve_node_azimuth_detailed(unstated)

    assert not resolution.used_catalogue_value
    assert resolution.is_display_normalisation
    assert resolution.note == NODE_CONVENTION_NOT_RENDERABLE
    assert resolution.convention is NodeConvention.UNSPECIFIED
    # pi/2 is the normalisation - PA 0, which is North - not 0.7 converted.
    assert resolve_node_azimuth(unstated) == pytest.approx(np.pi / 2.0)
    assert resolve_node_azimuth(unstated) != pytest.approx(np.pi / 2.0 - 0.7)


def test_a_stated_convention_is_used_and_converted():
    """The gate is a gate: a recorded convention makes the number usable."""
    stated = replace(
        measured(0.7, u.rad, provenance="test"),
        extra={"node_convention": NodeConvention.PA_EAST_OF_NORTH_RECEDING.value},
    )
    resolution = resolve_node_azimuth_detailed(stated)

    assert resolution.used_catalogue_value
    assert resolution.note == ""
    assert resolution.azimuth_rad == pytest.approx(np.pi / 2.0 - 0.7)


def test_display_normalisation_may_still_assume_the_standard_convention():
    """It invented the number, so it is entitled to say what it means.

    ``Omega_PA = 0`` written by ``for_display`` means North, and North is
    pi/2 internally. Refusing to convert it would draw the line of nodes
    East while the caption beside it said North.
    """
    normalised = assumed(0.0, u.rad, provenance="display-normalisation")
    resolution = resolve_node_azimuth_detailed(normalised)

    assert resolution.used_catalogue_value
    assert resolution.azimuth_rad == pytest.approx(np.pi / 2.0)

    # And an absent node uses the caller's default position angle, converted.
    assert resolve_node_azimuth(None) == pytest.approx(np.pi / 2.0)
    assert resolve_node_azimuth(unknown(u.rad)) == pytest.approx(np.pi / 2.0)


def test_an_unrenderable_node_is_relabelled_so_the_guide_is_not_called_measured():
    """The angle drawn and the word beside it have to agree.

    Leaving the element MEASURED while drawing the normalisation would put a
    solid line of nodes on screen with a caption calling it an observation.
    """
    from astro_explorer.physics.orbital_elements import OrbitalElements

    elements = OrbitalElements(
        name="probe b",
        semimajor_axis=measured(1.0, u.au, provenance="test"),
        eccentricity=measured(0.1, provenance="test"),
        period=measured(365.0, u.day, provenance="test"),
        longitude_of_ascending_node=measured(0.7, u.rad, provenance="test"),
    )
    assert elements.longitude_of_ascending_node.status is Status.MEASURED

    display = elements.for_display()
    shown = display.longitude_of_ascending_node
    assert shown.status is Status.ASSUMED_FOR_VISUALIZATION
    assert shown.value == pytest.approx(0.0)
    assert "convention was not recorded" in shown.note

    # The raw element is untouched: the catalogue value is not destroyed.
    assert elements.longitude_of_ascending_node.value == pytest.approx(0.7)


def test_the_overlay_caption_matches_the_line_it_is_drawn_beside():
    """A dashed guide must not carry a caption calling it an observation.

    The overlay decides its style from the *displayed* element and used to
    write its caption from the *published* one. Before C3.6 those could not
    disagree; the hardening makes them disagree for exactly one case, so the
    caption now reads the same source the guide does.
    """
    from astro_explorer.coordinates.system_frame import SystemFrame
    from astro_explorer.physics.orbital_elements import OrbitalElements
    from astro_explorer.rendering.renderer import GuideStyle
    from astro_explorer.rendering.scene_builder import orientation_guides

    def elements_with(node):
        return OrbitalElements(
            name="probe b",
            semimajor_axis=measured(1.0, u.au, provenance="test"),
            eccentricity=measured(0.2, provenance="test"),
            period=measured(365.0, u.day, provenance="test"),
            inclination=measured(np.deg2rad(40.0), u.rad, provenance="test"),
            longitude_of_ascending_node=node,
        )

    frame = SystemFrame.for_host("probe")
    unstated = measured(np.deg2rad(120.0), u.rad, provenance="test")
    stated = replace(
        unstated,
        extra={"node_convention": NodeConvention.PA_EAST_OF_NORTH_RECEDING.value},
    )

    # Stated: drawn at its own value, solid, and called measured.
    solid = orientation_guides(elements_with(stated), frame)
    node_guide = solid.guide("ascending-node")
    assert node_guide is not None
    assert node_guide.style is GuideStyle.SOLID
    assert any("Ascending node: 120 deg (measured)" in a for a in solid.annotations)

    # Unstated: not drawn at all unless the normalisation is asked for, and
    # the caption says the value was published and could not be used.
    withheld = orientation_guides(elements_with(unstated), frame)
    assert withheld.guide("ascending-node") is None
    assert any(
        "convention it was measured under was not recorded" in a
        for a in withheld.annotations
    )
    assert any("120 deg was published" in a for a in withheld.annotations)

    shown = orientation_guides(
        elements_with(unstated), frame, show_normalised=True
    )
    normalised = shown.guide("ascending-node")
    assert normalised is not None
    assert normalised.style is GuideStyle.DASHED
    assert not any("(measured)" in a for a in shown.annotations if "node" in a.lower())


# ==========================================================================
# Online refresh, offline runtime
# ==========================================================================


def _sample_record(source_id: str = "1111111111111111111") -> GaiaAstrometryRecord:
    return GaiaAstrometryRecord(
        source_id=source_id,
        ref_epoch=GAIA_DR3_REFERENCE_EPOCH_JYEAR,
        ra=10.0,
        ra_error=0.02,
        dec=20.0,
        dec_error=0.02,
        parallax=25.0,
        parallax_error=0.03,
        pmra=100.0,
        pmra_error=0.03,
        pmdec=-50.0,
        pmdec_error=0.03,
        radial_velocity=12.0,
        radial_velocity_error=0.5,
        astrometric_params_solved=31,
        correlations={"ra_dec_corr": -0.1},
    )


def test_the_online_sync_writes_the_cache_atomically(tmp_path):
    """The new file is renamed into place, never written over the old one.

    A crash or a full disk mid-write must leave the previous cache intact
    rather than a truncated document that happens to parse.
    """
    path = tmp_path / "gaia.json"
    cache = GaiaAstrometryCache(path)

    result = cache.refresh(["1111111111111111111"], fetcher=lambda ids: [_sample_record()])
    assert result.outcome == "committed"
    assert path.exists()
    assert not (tmp_path / "gaia.json.tmp").exists()

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["sha256"] == result.checksum
    assert set(document["sources"]) == {"1111111111111111111"}

    # The mechanism is the rename, not a lucky ordering of writes.
    source = (SRC / "data" / "gaia.py").read_text(encoding="utf-8")
    assert "os.replace(" in source


def test_a_replace_that_fails_leaves_the_previous_cache_intact(tmp_path, monkeypatch):
    """The failure is simulated at the last possible step."""
    import astro_explorer.data.gaia as gaia_module

    path = tmp_path / "gaia.json"
    cache = GaiaAstrometryCache(path)
    cache.refresh(["1111111111111111111"], fetcher=lambda ids: [_sample_record()])
    before = path.read_text(encoding="utf-8")

    def explode(source, target):
        raise OSError("no space left on device")

    monkeypatch.setattr(gaia_module.os, "replace", explode)
    with pytest.raises(OSError):
        cache.refresh(
            ["2222222222222222222"], fetcher=lambda ids: [_sample_record("2222222222222222222")]
        )

    assert path.read_text(encoding="utf-8") == before


def test_a_failed_refresh_preserves_the_previous_cache(tmp_path):
    """Three ways a refresh can fail; none of them costs the old data."""
    path = tmp_path / "gaia.json"
    cache = GaiaAstrometryCache(path)
    cache.refresh(["1111111111111111111"], fetcher=lambda ids: [_sample_record()])
    before = path.read_text(encoding="utf-8")

    def offline(ids):
        raise ConnectionError("the TAP service is unreachable")

    result = cache.refresh(["1111111111111111111"], fetcher=offline)
    assert result.outcome == "download_failed"
    assert not result.succeeded
    assert "ConnectionError" in result.detail
    assert path.read_text(encoding="utf-8") == before

    # An empty response is a failure, not an instruction to empty the cache.
    result = cache.refresh(["1111111111111111111"], fetcher=lambda ids: [])
    assert result.outcome == "rejected"
    assert path.read_text(encoding="utf-8") == before

    # And a response that fails validation is rejected wholesale.
    def malformed(ids):
        return [replace(_sample_record(), ref_epoch=1991.25)]

    result = cache.refresh(["1111111111111111111"], fetcher=malformed)
    assert result.outcome == "rejected"
    assert result.rejected
    assert path.read_text(encoding="utf-8") == before

    # The cache is still usable, not merely still present.
    assert cache.state_for("1111111111111111111") is not None


def test_a_source_that_was_not_requested_is_refused(tmp_path):
    """An identity failure beats a plausible nearby wrong star.

    A row the query did not ask for means the match was made some other
    way, and caching it would file one star's astrometry under another
    host's name - which is exactly the outcome using the archive's own
    cross-match was meant to avoid.
    """
    path = tmp_path / "gaia.json"
    cache = GaiaAstrometryCache(path)
    cache.refresh(["1111111111111111111"], fetcher=lambda ids: [_sample_record()])
    before = path.read_text(encoding="utf-8")

    def wrong_star(ids):
        return [_sample_record("9999999999999999999")]

    result = cache.refresh(["1111111111111111111"], fetcher=wrong_star)
    assert result.outcome == "rejected"
    assert "not requested" in result.detail
    assert path.read_text(encoding="utf-8") == before


def test_a_duplicated_source_is_refused(tmp_path):
    """``gaia_source`` is keyed on source_id, so two rows is not an answer.

    Choosing one of them would be a coin toss between two astrometric
    solutions for the same star.
    """
    path = tmp_path / "gaia.json"
    cache = GaiaAstrometryCache(path)

    def twice(ids):
        return [_sample_record(), _sample_record()]

    result = cache.refresh(["1111111111111111111"], fetcher=twice)
    assert result.outcome == "rejected"
    assert "more than once" in result.detail
    assert not path.exists()


def test_a_requested_source_missing_from_the_release_is_not_an_error(tmp_path):
    """A host genuinely absent from DR3 has no astrometry, and that is fine.

    The rest of the batch must still be cached: one uncross-matched host is
    not a reason to leave five others without a reference epoch.
    """
    path = tmp_path / "gaia.json"
    cache = GaiaAstrometryCache(path)

    result = cache.refresh(
        ["1111111111111111111", "2222222222222222222"],
        fetcher=lambda ids: [_sample_record()],
    )
    assert result.outcome == "committed"
    assert result.written == 1
    assert cache.record_for("1111111111111111111") is not None
    assert cache.record_for("2222222222222222222") is None


def test_there_is_no_cone_search_fallback():
    """An exact identifier that fails must not be replaced by a guess.

    Checked structurally, because the tempting repair - "the id did not
    resolve, look near the coordinates" - is most confident exactly where it
    is most likely wrong: a crowded field around a high-proper-motion star
    at an epoch we cannot state.
    """
    source = (SRC / "data" / "gaia.py").read_text(encoding="utf-8")
    code = _code_only(SRC / "data" / "gaia.py")

    for forbidden in ("CONTAINS", "CIRCLE", "POINT", "DISTANCE("):
        assert forbidden not in code, forbidden
    # The refusal is documented where someone would go looking to add one.
    assert "cone-search fallback" in source
    assert "GaiaIdentityError" in code


def test_a_cache_with_an_unknown_schema_is_refused(tmp_path):
    """Fail closed on a document this build does not understand.

    Field names are stable across Gaia releases and their meanings are not,
    so a best-effort read of an unrecognised document would propagate from
    the wrong reference epoch without a symptom.
    """
    path = tmp_path / "gaia.json"
    cache = GaiaAstrometryCache(path)
    cache.refresh(["1111111111111111111"], fetcher=lambda ids: [_sample_record()])

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema_version"] == CACHE_SCHEMA_VERSION

    document["schema_version"] = CACHE_SCHEMA_VERSION + 1
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="cache schema"):
        cache.records()


def test_a_cache_from_a_different_release_is_refused(tmp_path):
    """DR2 and DR3 share column names and differ in reference epoch."""
    path = tmp_path / "gaia.json"
    cache = GaiaAstrometryCache(path)
    cache.refresh(["1111111111111111111"], fetcher=lambda ids: [_sample_record()])

    document = json.loads(path.read_text(encoding="utf-8"))
    document["release"] = "Gaia DR2"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="Gaia DR2"):
        cache.records()


def test_the_committed_cache_declares_its_schema_and_release(gaia_cache):
    """The guards above are only worth having if the shipped file passes."""
    document = gaia_cache.load()
    assert document["schema_version"] == CACHE_SCHEMA_VERSION
    assert document["release"] == GAIA_DR3_RELEASE
    assert document["time_scale"] == GAIA_DR3_TIME_SCALE
    assert document["reference_epoch_jyear"] == GAIA_DR3_REFERENCE_EPOCH_JYEAR


def test_the_reference_epoch_is_a_real_tcb_time_object(gaia_cache):
    """Not reduced to "validated == True" and then discarded.

    Every accepted row equals the DR3 constant, which makes it tempting to
    store a flag and reconstruct the epoch from a literal later. The state
    has to carry the instant itself, in Gaia's own time scale, because that
    is what Astropy propagates from.
    """
    for source_id in gaia_cache.records():
        state = gaia_cache.state_for(source_id)
        epoch = state.reference_epoch
        assert isinstance(epoch, Time), source_id
        assert epoch.scale == "tcb", source_id
        assert epoch.jyear == pytest.approx(2016.0, abs=1e-9), source_id
        # The same instant, stated the other way round.
        assert epoch.jd == pytest.approx(
            Time(2016.0, format="jyear", scale="tcb").jd, abs=1e-9
        )


def test_an_unchanged_refresh_does_not_rewrite_the_cache(tmp_path):
    """Identical remote data is an outcome, not a commit."""
    path = tmp_path / "gaia.json"
    cache = GaiaAstrometryCache(path)
    cache.refresh(["1111111111111111111"], fetcher=lambda ids: [_sample_record()])
    stamped = json.loads(path.read_text(encoding="utf-8"))["retrieved"]

    result = cache.refresh(["1111111111111111111"], fetcher=lambda ids: [_sample_record()])
    assert result.outcome == "unchanged"
    assert result.succeeded
    assert json.loads(path.read_text(encoding="utf-8"))["retrieved"] == stamped


def test_the_cache_round_trips_every_preserved_field(tmp_path):
    """Uncertainties and correlations survive, because re-querying needs a net."""
    path = tmp_path / "gaia.json"
    cache = GaiaAstrometryCache(path)
    original = _sample_record()
    cache.refresh(["1111111111111111111"], fetcher=lambda ids: [original])

    restored = cache.record_for("1111111111111111111")
    assert restored == original
    assert restored.correlations == {"ra_dec_corr": -0.1}


def test_the_committed_cache_keeps_the_correlation_coefficients(gaia_cache):
    """RA, Dec, parallax and both proper motions are jointly fitted.

    The covariance is not reconstructible from five diagonal errors, so a
    cache that stored only the values would have to be rebuilt from the
    network the day anything propagates uncertainty.
    """
    for source_id, record in gaia_cache.records().items():
        assert record.ra_error is not None, source_id
        assert record.parallax_error is not None, source_id
        assert "ra_dec_corr" in record.correlations, source_id
        assert "pmra_pmdec_corr" in record.correlations, source_id


def test_normal_runtime_performs_no_network_access(monkeypatch, catalog):
    """Startup, propagation and inspection with the network forbidden.

    Not "does not need" - *cannot*. Every socket construction raises, so any
    live request anywhere under this call fails the test rather than merely
    being slow.
    """
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("the runtime path must not open a socket")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)

    for host in ("HD 80606", "HD 219134", "TRAPPIST-1", "Kepler-11"):
        system = build_slice(host, catalog)
        assert system.astrometry is not None, host
        assert system.astrometry.reference_epoch is not None, host

        propagated = system.host_astrometry(EPOCH_JD)
        assert propagated is not None
        record = system.planets[0]
        assert system.inspect_planet(record, EPOCH_JD)


def test_the_gaia_module_reaches_the_network_only_from_the_refresh_path():
    """One function downloads; nothing the application calls does.

    Checked structurally rather than by trusting the docstring: ``requests``
    may be imported only inside the fetcher.
    """
    source = (SRC / "data" / "gaia.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    importers = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            if any(str(n).split(".")[0] == "requests" for n in names):
                importers.append(node)

    assert importers, "the fetcher should import requests somewhere"
    functions = [
        f
        for f in ast.walk(tree)
        if isinstance(f, ast.FunctionDef)
        and any(n in ast.walk(f) for n in importers)
    ]
    assert {f.name for f in functions} == {"fetch_gaia_astrometry"}


def test_the_offline_check_in_ci_exercises_the_astrometric_state():
    """A green offline step that never touched astrometry would say nothing."""
    yaml = pytest.importorskip("yaml")
    workflow_path = ROOT / ".github" / "workflows" / "ci.yml"
    if not workflow_path.exists():  # pragma: no cover
        pytest.skip("CI workflow not present")

    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["tests"]["steps"]
    offline = next(s for s in steps if "Offline" in s.get("name", ""))

    assert "host_astrometry" in offline["run"]
    assert "reference_epoch" in offline["run"]
    assert "socket" in offline["run"]

    names = [s.get("name", "") for s in steps]
    assert any("C3.6" in name for name in names)


# ==========================================================================
# The architecture line holds
# ==========================================================================


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = path.relative_to(SRC).parts[:-1]
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = ["astro_explorer", *package]
                trimmed = base[: len(base) - node.level + 1]
                modules.add(".".join(trimmed + ([node.module] if node.module else [])))
            elif node.module:
                modules.add(node.module)
    return modules


def test_the_renderer_imports_no_astrometric_science():
    """The renderer draws; it does not decide where a star is.

    An astrometric position that reached the rendering layer would be one
    the scene could disagree with, and the scene works in float32 with
    exaggerated radii - the exact reasons C3 forbade measuring off it.
    """
    for path in sorted((SRC / "rendering").rglob("*.py")):
        for module in _imported_modules(path):
            assert "astrometry" not in module, path.name
            assert not module.startswith("astro_explorer.data"), path.name


def test_the_astrometry_module_stays_in_the_coordinates_layer():
    """It may use Astropy and provenance; it may not use data or rendering."""
    modules = _imported_modules(SRC / "coordinates" / "astrometry.py")
    for module in modules:
        assert not module.startswith("astro_explorer.data"), module
        assert not module.startswith("astro_explorer.rendering"), module
        assert not module.startswith("astro_explorer.ui"), module
    assert any(m.startswith("astropy") for m in modules)


def test_the_space_motion_model_is_stated_on_every_propagated_value():
    """An approximation that is not named is one nobody can argue with."""
    state = _high_proper_motion()
    result = propagate_astrometry(state, astropy_time(EPOCH_JD))

    assert "rectilinear" in result.position.ra.note
    assert "apply_space_motion" in result.position.ra.note
    assert any("rectilinear" in line for line in result.describe())
