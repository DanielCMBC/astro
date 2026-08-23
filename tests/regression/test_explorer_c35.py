"""Explorer C3.5: the SystemFrame -> ICRS basis.

C3 withheld the planet's absolute celestial position, and was right to.
The host's vector is ICRS; the planet's is in the system frame; adding them
adds components measured along different axes. That is a basis error, and
provenance labelling does not repair it.

This slice supplies the rotation that was missing, which means it is
almost entirely a file about conventions. The arithmetic is four lines. The
risk is that the four lines encode a *different* convention from the one
the catalogue used, and produce orbits that are mirrored, rotated ninety
degrees, or reflected through the plane of the sky - all of which look
completely plausible in a render.

So the tests here do two things the implementation cannot do for itself:

* check the triad against **Astropy**, by finite-differencing position with
  respect to RA and Dec, rather than against the same formulas restated;
* pin the **line-of-sight sign** explicitly, because that is the classic
  place a silent 180-degree error lives and it cannot be caught by any
  round-trip test - a wrong sign round-trips perfectly.

The last section is the part that matters scientifically: having a valid
rotation does not license publishing an absolute position. When Omega was
never measured, the rotation is exact and its input is a convention.
"""

from __future__ import annotations

from dataclasses import replace

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from astro_explorer.app.vertical_slice import build_slice, load_reference_catalog
from astro_explorer.coordinates.frames import Frame, SkyPosition, sky_position
from astro_explorer.coordinates.inspector import (
    ABSOLUTE_POSITION_NO_HOST,
    ABSOLUTE_POSITION_NO_ORBIT,
    InspectorFrame,
    absolute_planet_position,
    planet_to_star_distance,
)
from astro_explorer.coordinates.tangent import (
    EPOCH_NOT_MODELLED,
    LINE_OF_SIGHT,
    NODE_CONVENTION_UNSTATED,
    NODE_SENSE_UNRESOLVED,
    POLE_DEGENERATE,
    POLE_TOLERANCE_DEG,
    SKY_BASIS_CONVENTION,
    NodeConvention,
    NodeSense,
    absolute_position_blockers,
    azimuth_to_position_angle,
    is_pole_degenerate,
    node_convention_of,
    node_sense_of,
    position_angle_to_azimuth,
    system_offset_to_icrs_pc,
    system_to_icrs_rotation,
    tangent_basis,
)
from astro_explorer.provenance import Status, measured, unknown

EPOCH_JD = 2460000.0

#: A deliberately awkward spread: both hemispheres, near the pole, near the
#: RA wrap. A basis bug that only shows up at delta = 0 is still a bug.
SKY_SAMPLES = [
    (0.0, 0.0),
    (37.0, 23.0),
    (140.657, 50.604),
    (359.9, -5.0),
    (180.0, -89.0),
    (270.0, 89.5),
    (123.45, -67.89),
]


@pytest.fixture(scope="module")
def catalog():
    try:
        return load_reference_catalog()
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("reference snapshot not committed")


@pytest.fixture(scope="module")
def hd80606(catalog):
    return build_slice("HD 80606", catalog)


@pytest.fixture(scope="module")
def hd219134(catalog):
    return build_slice("HD 219134", catalog)


@pytest.fixture(scope="module")
def trappist1(catalog):
    """Detached: a good tangent basis, no origin to hang it on."""
    return build_slice("TRAPPIST-1", catalog)


# ==========================================================================
# The triad is a proper orthonormal basis, everywhere on the sky
# ==========================================================================


@pytest.mark.parametrize("ra,dec", SKY_SAMPLES)
def test_the_tangent_triad_is_orthonormal(ra, dec):
    basis = tangent_basis(ra, dec)

    for vector in (basis.east, basis.north, basis.radial):
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-15)
    assert basis.east @ basis.north == pytest.approx(0.0, abs=1e-15)
    assert basis.east @ basis.radial == pytest.approx(0.0, abs=1e-15)
    assert basis.north @ basis.radial == pytest.approx(0.0, abs=1e-15)

    # east x north = radial fixes the handedness, and is what makes +Z of
    # the canonical frame point away from the observer.
    assert np.allclose(np.cross(basis.east, basis.north), basis.radial, atol=1e-15)


@pytest.mark.parametrize("ra,dec", SKY_SAMPLES)
def test_the_rotation_is_proper_with_determinant_plus_one(ra, dec):
    """A determinant of -1 would be a reflection, not a rotation.

    It would preserve every length and angle this file checks and still
    mirror every orbit. Only the determinant catches it.
    """
    matrix = tangent_basis(ra, dec).matrix()

    assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-14)
    assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-14)
    assert tangent_basis(ra, dec).is_orthonormal()


@pytest.mark.parametrize("ra,dec", SKY_SAMPLES)
def test_the_radial_direction_is_the_host_direction(ra, dec):
    """e_r must be the unit vector Astropy puts the star along."""
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=10.0 * u.pc)
    expected = np.array(
        [
            coord.cartesian.x.to_value(u.pc),
            coord.cartesian.y.to_value(u.pc),
            coord.cartesian.z.to_value(u.pc),
        ]
    )
    expected /= np.linalg.norm(expected)

    assert np.allclose(tangent_basis(ra, dec).radial, expected, atol=1e-12)


@pytest.mark.parametrize("ra,dec", SKY_SAMPLES)
def test_east_and_north_agree_with_astropy_finite_differences(ra, dec):
    """The tangent directions are checked against Astropy, not against us.

    Restating the same closed-form expressions in the test would prove only
    that the file was copied correctly. Differencing an Astropy position
    with respect to RA and Dec is an independent derivation: east is the
    direction the star moves for increasing RA, north for increasing Dec.
    """
    if abs(dec) > 89.0:
        pytest.skip("the RA derivative degenerates at the pole")

    step = 1e-6

    def unit(ra_deg, dec_deg):
        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, distance=1.0 * u.pc)
        vector = np.array(
            [
                coord.cartesian.x.to_value(u.pc),
                coord.cartesian.y.to_value(u.pc),
                coord.cartesian.z.to_value(u.pc),
            ]
        )
        return vector / np.linalg.norm(vector)

    d_ra = (unit(ra + step, dec) - unit(ra - step, dec)) / (2 * step)
    d_dec = (unit(ra, dec + step) - unit(ra, dec - step)) / (2 * step)

    basis = tangent_basis(ra, dec)
    assert np.allclose(d_ra / np.linalg.norm(d_ra), basis.east, atol=1e-7)
    assert np.allclose(d_dec / np.linalg.norm(d_dec), basis.north, atol=1e-7)


@pytest.mark.parametrize("ra,dec", SKY_SAMPLES)
def test_the_rotation_round_trips(ra, dec):
    matrix = tangent_basis(ra, dec).matrix()
    rng = np.random.default_rng(11)

    for _ in range(5):
        vector = rng.normal(size=3)
        assert np.allclose(matrix.T @ (matrix @ vector), vector, atol=1e-14)
        # A rotation preserves length; that is what makes a distance
        # frame-independent.
        assert np.linalg.norm(matrix @ vector) == pytest.approx(
            np.linalg.norm(vector), rel=1e-14
        )


# ==========================================================================
# The convention: which axis is which, and which way the sky runs
# ==========================================================================


@pytest.mark.parametrize("ra,dec", SKY_SAMPLES)
def test_the_canonical_axes_are_east_north_and_away(ra, dec):
    basis = tangent_basis(ra, dec)
    matrix = basis.matrix()

    assert np.allclose(matrix @ np.array([1.0, 0.0, 0.0]), basis.east, atol=1e-15)
    assert np.allclose(matrix @ np.array([0.0, 1.0, 0.0]), basis.north, atol=1e-15)
    assert np.allclose(matrix @ np.array([0.0, 0.0, 1.0]), basis.radial, atol=1e-15)


@pytest.mark.parametrize("ra,dec", SKY_SAMPLES)
def test_the_line_of_sight_points_away_from_the_observer(ra, dec):
    """The 180-degree trap, pinned against the *ascending = receding* rule.

    A round trip cannot catch this: a flipped ``+Z`` round-trips perfectly
    and mirrors every orbit through the plane of the sky. Only a direct
    assertion catches it.

    An earlier draft of this module used ``+x = North, +y = East``, which is
    self-consistent and forces ``+z`` toward the observer - making this
    codebase's "ascending" node the approaching one, in silent disagreement
    with every catalogue it reads.
    """
    basis = tangent_basis(ra, dec)
    z_in_icrs = basis.matrix() @ np.array([0.0, 0.0, 1.0])

    # Away from the observer means along the radial direction, which runs
    # from the Sun outward to the star.
    assert z_in_icrs @ basis.radial == pytest.approx(1.0, abs=1e-14)
    assert "receding" in LINE_OF_SIGHT

    # A planet displaced along +Z is further from us than its host.
    host = 12.0 * basis.radial
    assert np.linalg.norm(host + 0.001 * z_in_icrs) > np.linalg.norm(host)


@pytest.mark.parametrize("i_deg", [0.5, 30.0, 89.0, 90.0, 120.0, 179.5])
def test_a_body_just_past_the_ascending_node_is_receding(i_deg):
    """The rule that fixes the sign of +Z, checked against the propagator.

    Under ``R_z R_x(i) R_z(omega)`` a body just past the node has
    ``z = sin(u) sin(i) > 0`` for any inclination, prograde or retrograde.
    The standard definition says that crossing is the *ascending* one, which
    is only true if ``+Z`` points away from the observer.
    """
    from astro_explorer.physics.orientation import rotation_x, rotation_z

    inclination = np.deg2rad(i_deg)
    for azimuth in (0.0, 1.0, 2.5):
        just_past = rotation_z(azimuth) @ rotation_x(inclination) @ np.array(
            [np.cos(1e-4), np.sin(1e-4), 0.0]
        )
        assert just_past[2] > 0.0, (i_deg, azimuth)


@pytest.mark.parametrize("ra,dec", SKY_SAMPLES)
def test_position_angle_zero_points_north_and_ninety_points_east(ra, dec):
    """PA is measured from North toward East; the internal azimuth is not.

    ``+X`` is East and ``+Y`` is North, so ``R_z`` carries East toward North
    - the opposite sense from the other axis. Feeding a catalogue angle
    straight into the rotation would put every node 90 degrees out and
    running backwards.
    """
    
    from astro_explorer.physics.orientation import rotation_z

    basis = tangent_basis(ra, dec)
    matrix = basis.matrix()

    def node_direction(position_angle_deg):
        theta = position_angle_to_azimuth(np.deg2rad(position_angle_deg))
        return matrix @ (rotation_z(theta) @ np.array([1.0, 0.0, 0.0]))

    assert np.allclose(node_direction(0.0), basis.north, atol=1e-12)
    assert np.allclose(node_direction(90.0), basis.east, atol=1e-12)
    assert np.allclose(node_direction(180.0), -basis.north, atol=1e-12)
    assert np.allclose(node_direction(270.0), -basis.east, atol=1e-12)


@pytest.mark.parametrize("pa_deg", [0.0, 17.0, 90.0, 133.7, 270.0, 359.0])
def test_the_position_angle_conversion_is_its_own_inverse(pa_deg):
    pa = np.deg2rad(pa_deg)
    assert azimuth_to_position_angle(position_angle_to_azimuth(pa)) == pytest.approx(pa)


def test_a_raw_position_angle_is_not_a_usable_azimuth():
    """The conversion is not a no-op, which is why it must be explicit."""
    for pa_deg in (10.0, 200.0, 300.0):
        pa = np.deg2rad(pa_deg)
        assert position_angle_to_azimuth(pa) != pytest.approx(pa)
    # The two senses coincide only at 45 degrees.
    assert position_angle_to_azimuth(np.deg2rad(45.0)) == pytest.approx(
        np.deg2rad(45.0)
    )


def test_the_convention_is_documented_in_one_place():
    """The words and the arithmetic must not be able to drift apart."""
    assert "+X = East" in SKY_BASIS_CONVENTION
    assert "+Y = North" in SKY_BASIS_CONVENTION
    assert "away from the observer" in SKY_BASIS_CONVENTION
    assert "receding" in SKY_BASIS_CONVENTION
    assert "pi/2 - Omega_PA" in SKY_BASIS_CONVENTION


# ==========================================================================
# Offsets land where the convention says they should
# ==========================================================================


def _probe(ra=37.0, dec=23.0, distance_pc=10.0) -> SkyPosition:
    return sky_position("probe", ra, dec, catalog_distance_pc=distance_pc)


def _tagged(node, *, convention=None, sense=None, evidence=None):
    """Attach node metadata the way an ingestion path would.

    The convention and sense live in ``Parameter.extra`` so a catalogue
    reader can record them without every consumer having to know, and so
    their *absence* is the default rather than something to opt out of.
    """
    extra = dict(node.extra)
    if convention is not None:
        extra["node_convention"] = convention.value
    if sense is not None:
        extra["node_sense"] = sense.value
    if evidence is not None:
        extra["node_sense_evidence"] = evidence
    return replace(node, extra=extra)


def _resolved_node(value_rad=0.7):
    """A node that clears every scientific gate except the epoch."""
    return _tagged(
        measured(value_rad, u.rad, provenance="test: RV-resolved"),
        convention=NodeConvention.PA_EAST_OF_NORTH_RECEDING,
        sense=NodeSense.RESOLVED,
        evidence="test: radial-velocity orbit identifies the receding node",
    )


@pytest.mark.parametrize("axis,expected", [(0, "east"), (1, "north"), (2, "away")])
def test_a_one_au_offset_follows_the_axis_it_was_given(axis, expected):
    """Each canonical axis must land on its own direction, alone."""
    host = _probe()
    basis = tangent_basis(37.0, 23.0)
    offset_au = np.zeros(3)
    offset_au[axis] = 1.0

    moved = system_offset_to_icrs_pc(host, offset_au)
    one_au_pc = float((1.0 * u.au).to_value(u.pc))
    directions = {"east": basis.east, "north": basis.north, "away": basis.radial}

    assert np.allclose(moved, directions[expected] * one_au_pc, atol=1e-18)
    assert np.linalg.norm(moved) == pytest.approx(one_au_pc, rel=1e-14)
    for name, other in directions.items():
        if name != expected:
            assert moved @ other == pytest.approx(0.0, abs=1e-18)


def test_a_plus_z_offset_moves_the_planet_further_from_the_sun():
    """+Z is away, so a receding planet is further, not nearer."""
    host = _probe(distance_pc=10.0)
    moved = system_offset_to_icrs_pc(host, np.array([0.0, 0.0, 1.0]))
    one_au_pc = float((1.0 * u.au).to_value(u.pc))

    host_pc = host.cartesian_pc(Frame.ICRS)
    assert np.linalg.norm(host_pc + moved) == pytest.approx(10.0 + one_au_pc, rel=1e-12)


def test_the_offset_is_carried_in_float64_not_float32():
    """One AU against ten parsecs is ~5e-7 relative: float32 loses it."""
    host = _probe(distance_pc=10.0)
    moved = system_offset_to_icrs_pc(host, np.array([1.0, 0.0, 0.0]))

    host_pc = host.cartesian_pc(Frame.ICRS)
    assert np.linalg.norm(moved) == pytest.approx(
        float((1.0 * u.au).to_value(u.pc)), rel=1e-12
    )
    assert np.allclose(np.float32(host_pc) + np.float32(moved), np.float32(host_pc))


def test_an_unlocated_host_still_has_a_valid_rotation(trappist1):
    """Direction and origin are independent pieces of knowledge."""
    rotation = system_to_icrs_rotation(trappist1.star.position)

    assert rotation is not None
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-14)
    assert not trappist1.star.position.has_distance


def test_a_host_with_no_direction_has_no_rotation():
    nameless = SkyPosition(
        name="probe", ra=unknown(u.deg), dec=unknown(u.deg), distance=measured(5.0, u.pc)
    )
    assert system_to_icrs_rotation(nameless) is None
    assert system_to_icrs_rotation(None) is None


# ==========================================================================
# The pole is algebraically fine and physically meaningless
# ==========================================================================


@pytest.mark.parametrize("dec", [90.0, -90.0])
def test_the_exact_pole_is_flagged_degenerate(dec):
    """The triad stays orthonormal there, which is exactly the danger.

    At the pole right ascension is not a unique physical direction, so the
    closed form will happily hand back a basis whose azimuth is set by
    whatever RA was recorded for an object that has no meaningful one.
    """
    assert is_pole_degenerate(dec)
    # ... and the arithmetic gives no hint that anything is wrong.
    assert tangent_basis(0.0, dec).is_orthonormal()
    assert tangent_basis(123.0, dec).is_orthonormal()
    # Two different recorded RAs give two different sky-plane bases.
    assert not np.allclose(
        tangent_basis(0.0, dec).east, tangent_basis(123.0, dec).east
    )


def test_ordinary_declinations_are_not_flagged():
    for dec in (0.0, 45.0, -60.0, 89.9, -89.9):
        assert not is_pole_degenerate(dec)
    assert not is_pole_degenerate(None)
    # The tolerance is a hair, not a zone.
    assert is_pole_degenerate(90.0 - POLE_TOLERANCE_DEG / 2)


def test_a_pole_host_cannot_publish_an_absolute_position():
    """No unique physical position angle exists there, so none is claimed."""
    pole = sky_position("pole", 0.0, 90.0, catalog_distance_pc=10.0)
    reasons = absolute_position_blockers(pole, _resolved_node(), epoch_resolved=True)

    assert POLE_DEGENERATE in reasons

    row = absolute_planet_position(
        pole, np.array([1.0, 0.0, 0.0]), node=_resolved_node(), epoch_resolved=True
    )
    assert not row.is_known
    assert POLE_DEGENERATE in row.note


# ==========================================================================
# Node convention and node sense: a number is not a direction
# ==========================================================================


def test_an_unstated_node_convention_never_defaults_to_the_standard():
    """Silence must not unlock anything.

    A catalogued angle whose convention was not recorded is a number, not a
    direction on the sky. Assuming the common convention would be right most
    of the time, which is the worst possible failure mode.
    """
    bare = measured(0.7, u.rad, provenance="test")
    assert node_convention_of(bare) is NodeConvention.UNSPECIFIED
    assert node_convention_of(None) is NodeConvention.UNSPECIFIED
    assert not NodeConvention.UNSPECIFIED.is_stated

    sensed_only = _tagged(
        measured(0.7, u.rad, provenance="test"), sense=NodeSense.RESOLVED
    )
    reasons = absolute_position_blockers(_probe(), sensed_only, epoch_resolved=True)
    assert NODE_CONVENTION_UNSTATED in reasons


def test_an_unrecognised_convention_string_is_not_trusted():
    """A typo must fail closed, not fall through as valid."""
    mistyped = replace(
        measured(0.7, u.rad, provenance="test"),
        extra={"node_convention": "PA_NORTH_OF_EAST"},
    )
    assert node_convention_of(mistyped) is NodeConvention.UNSPECIFIED


def test_a_measured_node_is_only_modulo_180_by_default():
    """A measured *number* is not a resolved node.

    Relative astrometry commonly determines the node only modulo 180
    degrees: (omega, Omega) and (omega + pi, Omega - pi) project to the same
    orbit on the sky. Only radial-velocity or equivalent line-of-sight
    information breaks the tie, so having a value defaults to MODULO_180.
    """
    plain = measured(0.7, u.rad, provenance="test")
    assert node_sense_of(plain) is NodeSense.MODULO_180
    assert not node_sense_of(plain).is_resolved

    assert node_sense_of(unknown(u.rad)) is NodeSense.UNKNOWN
    assert node_sense_of(None) is NodeSense.UNKNOWN
    assert node_sense_of(_resolved_node()) is NodeSense.RESOLVED


def test_a_modulo_180_node_does_not_unlock_an_absolute_position():
    """The publication rule, at its most important boundary."""
    stated_but_ambiguous = _tagged(
        measured(0.7, u.rad, provenance="test"),
        convention=NodeConvention.PA_EAST_OF_NORTH_RECEDING,
        sense=NodeSense.MODULO_180,
    )
    reasons = absolute_position_blockers(
        _probe(), stated_but_ambiguous, epoch_resolved=True
    )
    assert NODE_SENSE_UNRESOLVED in reasons

    row = absolute_planet_position(
        _probe(),
        np.array([0.3, 0.0, 0.0]),
        node=stated_but_ambiguous,
        epoch_resolved=True,
    )
    assert not row.is_known
    assert NODE_SENSE_UNRESOLVED in row.note


def test_a_resolved_node_sense_does_unlock_it():
    """The gate is a gate, not a wall: RV-resolved information opens it."""
    reasons = absolute_position_blockers(_probe(), _resolved_node(), epoch_resolved=True)
    assert reasons == []

    row = absolute_planet_position(
        _probe(), np.array([0.3, -0.1, 0.05]), node=_resolved_node(), epoch_resolved=True
    )
    assert row.is_known
    assert row.frame is InspectorFrame.ICRS
    assert row.status is not Status.ASSUMED_FOR_VISUALIZATION
    assert row.note == ""


def test_the_projected_degeneracy_is_real():
    """(omega, Omega) and (omega + pi, Omega - pi) project identically.

    This is *why* a measured node is only modulo 180. The two orientations
    put the orbit in the same place on the sky and differ only in which node
    is receding - which is invisible without line-of-sight information.
    """
    from astro_explorer.physics.orientation import rotation_perifocal_to_inertial

    inclination = np.deg2rad(63.0)
    omega, node = np.deg2rad(41.0), np.deg2rad(112.0)

    original = rotation_perifocal_to_inertial(inclination, omega, node)
    flipped = rotation_perifocal_to_inertial(inclination, omega + np.pi, node - np.pi)

    perifocal = np.array([0.7, -0.3, 0.0])
    a = original @ perifocal
    b = flipped @ perifocal

    # Same projection on the sky (X and Y), opposite line-of-sight sign.
    assert a[0] == pytest.approx(b[0], abs=1e-12)
    assert a[1] == pytest.approx(b[1], abs=1e-12)
    assert a[2] == pytest.approx(-b[2], abs=1e-12)


# ==========================================================================
# The coordinate epoch gate, which currently blocks everything real
# ==========================================================================


def _absolute_row(system, record):
    return next(
        r
        for r in system.inspect_planet(record, EPOCH_JD)
        if r.label == "Absolute position"
    )


def test_the_epoch_gate_blocks_by_default():
    """SkyPosition has no obstime, proper motion or radial velocity.

    So a host position is at its catalogue epoch while the planet offset is
    at the requested time. For a nearby high-proper-motion star that
    mismatch is a larger physical error than the AU-scale offset it would be
    added to, which is why it blocks rather than being ignored as small.
    """
    for field in ("obstime", "proper_motion", "pm_ra", "pm_dec", "radial_velocity"):
        assert field not in SkyPosition.__dataclass_fields__

    reasons = absolute_position_blockers(_probe(), _resolved_node())
    assert EPOCH_NOT_MODELLED in reasons


def test_a_real_planet_is_blocked_by_the_epoch_and_the_node(hd80606):
    """Both reasons are reported, not just the first one found."""
    record = hd80606.planet("HD 80606 b")
    row = _absolute_row(hd80606, record)

    assert not row.is_known
    assert NODE_SENSE_UNRESOLVED in row.note
    assert EPOCH_NOT_MODELLED in row.note


def test_no_real_system_publishes_an_absolute_position(hd80606, hd219134, trappist1):
    """The whole snapshot, swept: nothing gets through the gates."""
    for system in (hd80606, hd219134, trappist1):
        for record in system.planets:
            row = _absolute_row(system, record)
            assert not row.is_known, (system.star.name, record.name)
            assert row.note


def test_a_detached_host_reports_every_blocking_reason(trappist1):
    record = trappist1.planets[0]
    state = trappist1.state(record, EPOCH_JD)
    row = absolute_planet_position(
        trappist1.star.position,
        state.position,
        node=record.elements.longitude_of_ascending_node,
    )

    assert not row.is_known
    assert ABSOLUTE_POSITION_NO_HOST in row.note
    assert EPOCH_NOT_MODELLED in row.note


def test_an_unpropagatable_orbit_says_so():
    row = absolute_planet_position(_probe(), None, node=_resolved_node())
    assert not row.is_known
    assert ABSOLUTE_POSITION_NO_ORBIT in row.note


def test_the_normalised_realisation_carries_every_unresolved_reason():
    """Available for drawing, never promoted to a catalogue coordinate."""
    row = absolute_planet_position(
        _probe(),
        np.array([0.3, -0.1, 0.05]),
        node=measured(0.7, u.rad, provenance="test"),
        normalised=True,
    )

    assert row.is_known
    assert row.status is Status.ASSUMED_FOR_VISUALIZATION
    assert NODE_SENSE_UNRESOLVED in row.note
    assert EPOCH_NOT_MODELLED in row.note


def test_a_normalised_realisation_still_needs_an_origin_and_an_offset():
    """Two things it cannot invent, however illustrative it may be."""
    detached = sky_position("probe", 12.0, 3.0, parallax_mas=-1.0)
    assert not absolute_planet_position(
        detached, np.array([1.0, 0.0, 0.0]), node=_resolved_node(), normalised=True
    ).is_known
    assert not absolute_planet_position(
        _probe(), None, node=_resolved_node(), normalised=True
    ).is_known


# ==========================================================================
# Planet -> selected star distance, gated the same way
# ==========================================================================


def test_the_planet_to_star_distance_needs_a_common_epoch(hd219134):
    """It is blocked by the epoch gate exactly as the position is."""
    host = _probe(distance_pc=10.0)
    result = planet_to_star_distance(
        host,
        np.array([0.3, -0.1, 0.05]),
        hd219134.star.position,
        node=_resolved_node(),
    )
    assert not result.is_known
    assert EPOCH_NOT_MODELLED in result.note


def test_the_planet_to_star_distance_works_once_every_gate_is_open(hd219134):
    host = _probe(distance_pc=10.0)
    other = hd219134.star.position
    offset_au = np.array([0.3, -0.1, 0.05])

    distance = planet_to_star_distance(
        host, offset_au, other, node=_resolved_node(), epoch_resolved=True
    )
    assert distance.is_known

    planet = absolute_planet_position(
        host, offset_au, node=_resolved_node(), epoch_resolved=True
    )
    expected = np.linalg.norm(other.cartesian_pc(Frame.ICRS) - planet.values)
    assert distance.value_in(u.pc) == pytest.approx(expected, rel=1e-14)

    # The AU-scale term is carried, not dropped for being small.
    star_to_star = np.linalg.norm(
        other.cartesian_pc(Frame.ICRS) - host.cartesian_pc(Frame.ICRS)
    )
    assert distance.value_in(u.pc) != star_to_star
    assert distance.value_in(u.pc) == pytest.approx(star_to_star, rel=1e-5)


def test_a_blocked_planet_to_star_distance_is_not_the_star_to_star_one(
    hd80606, hd219134
):
    """No silent substitution of a different question's answer.

    Falling back would be plausible - the two differ by an AU at parsec
    range - and the number would carry nothing saying it was about the star.
    """
    record = hd80606.planet("HD 80606 b")
    state = hd80606.state(record, EPOCH_JD)

    result = planet_to_star_distance(
        hd80606.star.position,
        state.position,
        hd219134.star.position,
        node=record.elements.longitude_of_ascending_node,
    )

    assert not result.is_known
    assert result.value is None
    star_to_star = np.linalg.norm(
        hd219134.star.position.cartesian_pc(Frame.ICRS)
        - hd80606.star.position.cartesian_pc(Frame.ICRS)
    )
    assert result.value != star_to_star


def test_a_normalised_position_does_not_leak_into_a_published_distance():
    """An illustrative position must not become a quotable separation."""
    host = _probe(distance_pc=10.0)
    other = sky_position("other", 200.0, -14.0, catalog_distance_pc=25.0)

    result = planet_to_star_distance(
        host,
        np.array([0.3, 0.0, 0.0]),
        other,
        node=measured(0.7, u.rad, provenance="test"),
    )
    assert not result.is_known


def test_an_unlocated_other_star_blocks_the_distance(trappist1):
    result = planet_to_star_distance(
        _probe(),
        np.array([0.1, 0.0, 0.0]),
        trappist1.star.position,
        node=_resolved_node(),
        epoch_resolved=True,
    )
    assert not result.is_known
    assert "no usable distance" in result.note


# ==========================================================================
# Nothing here is downstream of the renderer
# ==========================================================================


def test_the_tangent_module_imports_no_rendering_primitives():
    import ast
    from pathlib import Path

    import astro_explorer.coordinates.tangent as tangent

    source = Path(tangent.__file__).read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert not any("rendering" in name for name in imported), imported
    assert not any(name.startswith("astro_explorer.data") for name in imported)


def test_the_rotation_is_invariant_under_display_settings(hd80606):
    """The C3 invariance rule, extended to the new transform."""
    from astro_explorer.rendering.scene_builder import build_frame_scene

    baseline = system_to_icrs_rotation(hd80606.star.position)
    anomalies = hd80606.mean_anomalies(EPOCH_JD)

    for exaggerate in (False, True):
        scene = build_frame_scene(
            hd80606.frame,
            hd80606.star,
            hd80606.planets,
            mean_anomalies=anomalies,
            exaggerate=exaggerate,
        )
        assert scene.planets
        assert np.array_equal(system_to_icrs_rotation(hd80606.star.position), baseline)


def test_every_c3_row_label_is_unchanged(hd80606):
    """C3.5 must not reshape the inspector, only qualify one of its rows."""
    record = hd80606.planet("HD 80606 b")
    labels = [r.label for r in hd80606.inspect_planet(record, EPOCH_JD)]

    assert labels == [
        "Distance from host",
        "Periapsis distance",
        "Apoapsis distance",
        "System-frame position",
        "Absolute position",
        "Phase provenance",
    ]


def test_a_resolved_tag_without_named_evidence_does_not_count():
    """RESOLVED is a claim about an observation, so it must name one.

    The tie between the two nodes is broken only by evidence specific to
    *this orbit* - a radial-velocity orbit, an eclipse timing that fixes
    which node recedes. A generic systemic stellar radial velocity is the
    thing most likely to be reached for and cannot do it: it describes the
    whole system's motion relative to the Sun and says nothing about which
    node of a planet's orbit is receding.

    Since that distinction is invisible in the number, an unevidenced
    RESOLVED tag falls back to MODULO_180 rather than unlocking publication.
    """
    unevidenced = _tagged(
        measured(0.7, u.rad, provenance="test"),
        convention=NodeConvention.PA_EAST_OF_NORTH_RECEDING,
        sense=NodeSense.RESOLVED,
    )
    assert node_sense_of(unevidenced) is NodeSense.MODULO_180

    blank = _tagged(
        measured(0.7, u.rad, provenance="test"),
        convention=NodeConvention.PA_EAST_OF_NORTH_RECEDING,
        sense=NodeSense.RESOLVED,
        evidence="   ",
    )
    assert node_sense_of(blank) is NodeSense.MODULO_180

    reasons = absolute_position_blockers(_probe(), unevidenced, epoch_resolved=True)
    assert NODE_SENSE_UNRESOLVED in reasons

    # Named evidence is what makes the difference.
    assert node_sense_of(_resolved_node()) is NodeSense.RESOLVED


def test_the_resolved_rule_names_the_systemic_velocity_trap():
    """The rule is written down where someone tagging data will read it."""
    doc = NodeSense.__doc__
    assert "systemic" in doc
    assert "not** enough" in doc
    assert "modulo 180" in doc
