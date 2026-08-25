"""Explorer C3.6: astrometric epoch and stellar space motion.

C3.5 supplied the SystemFrame -> ICRS rotation and then declined to publish
an absolute planet position anyway, for a reason that was left as the last
deliberately closed gate in the pipeline:

    :class:`~astro_explorer.coordinates.frames.SkyPosition` carries RA, Dec,
    distance and a frame, and nothing else. It has no reference epoch, no
    proper motion and no radial velocity, so a host's catalogue position is
    at *some* unstated instant while the planet offset is at the requested
    one.

C3.5 modelled that gate as a boolean, ``epoch_resolved``, defaulting to
False and never set True anywhere except in tests. That was honest as long
as it stayed a placeholder and dangerous the moment it did not: a bare
``True`` at one call site would have opened every downstream gate without
a single number changing anywhere. This module replaces the boolean with
the thing it was standing in for.

Why the mismatch is not a rounding error
----------------------------------------

The offset being added is astronomical-unit scale. The error being ignored
is proper motion times elapsed time, and for a nearby star it is much
larger. HD 219134 moves 2.1 arcsec per year and sits 6.5 pc away, so a
decade off epoch displaces it by about 0.14 mas at that distance - roughly
135 AU. Adding a 0.2 AU planet offset to a host position that is 135 AU
wrong is not a small inconsistency; it is an answer dominated entirely by
the term nobody modelled.

Three knowledge tiers, not one flag
-----------------------------------

:class:`MotionKnowledge` exists because "we know where this star is" has
three genuinely different meanings and collapsing them is how a direction
becomes a position:

``REFERENCE_EPOCH_ONLY``
    RA, Dec, distance and the epoch they were measured at. Complete and
    publishable **at that epoch** - and at that epoch no motion has to be
    invented, which is a real answer rather than a degraded one.

``DIRECTION_ONLY``
    Plus both proper-motion components. The star's *direction* can be moved
    to another date. Its distance cannot: without a radial velocity there
    is no rate of change along the line of sight. A propagated direction
    combined with a reference-epoch distance is not a cross-epoch 3D
    position, and this module refuses to call it one.

``FULL_SPACE_MOTION``
    Plus radial velocity. A uniform rectilinear space-motion model then
    determines the whole 3D state at any date, and Astropy evaluates it.

Missing is not zero
-------------------

A star with no published radial velocity does not have a radial velocity of
zero, and the difference is the whole point of this module: substituting
zero produces a confident, plausible, wrong distance at every epoch except
the reference one. Astropy's :meth:`~astropy.coordinates.SkyCoord.apply_space_motion`
will happily assume zero for a missing component, so nothing here hands it
a coordinate whose motion is incomplete and then reads the result as a 3D
position. An *explicitly measured* zero is a different thing entirely, and
survives as zero.

Astropy owns the propagation
----------------------------

The transformation itself is
:meth:`~astropy.coordinates.SkyCoord.apply_space_motion`. Hand-written
proper-motion propagation would have to get the ``cos(dec)`` convention,
the right-ascension wrap, the perspective acceleration and the radial
foreshortening all right, and the first three of those fail silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from ..physics.epoch import INSTANT_MATCH_TOLERANCE_DAYS
from ..provenance import Parameter, Status, derived, unknown
from .frames import Frame, SkyPosition

__all__ = [
    "MotionKnowledge",
    "AstrometricState",
    "PropagatedAstrometry",
    "propagate_astrometry",
    "at_reference_epoch",
    "EPOCH_MATCH_TOLERANCE_DAYS",
    "REFERENCE_EPOCH_NOT_STATED",
    "PROPER_MOTION_NOT_MEASURED",
    "RADIAL_VELOCITY_NOT_MEASURED",
    "DISTANCE_NOT_USABLE",
    "DIRECTION_ONLY_NOT_A_POSITION",
    "DIRECTION_ONLY_IS_MODEL_DEPENDENT",
    "ZERO_RV_APPROXIMATION",
    "SPACE_MOTION_MODEL",
]

#: The kinematic model a propagated state is produced under, stated because
#: it is an approximation and not a law: the star is assumed to move in a
#: straight line at constant velocity relative to the solar-system
#: barycentre. Real stars accelerate in the Galactic potential and orbit
#: unseen companions; over the decades this explorer spans, neither term is
#: resolvable, and over millennia both are.
SPACE_MOTION_MODEL = (
    "uniform rectilinear space motion relative to the solar-system "
    "barycentre, evaluated by astropy SkyCoord.apply_space_motion; no "
    "Galactic acceleration and no unseen-companion reflex is modelled"
)

#: Two epochs closer than this are the same epoch.
#:
#: Re-exported from :mod:`astro_explorer.physics.epoch` rather than defined
#: here, because "the host, the target star and the planet are at one time"
#: has to mean the same thing on the stellar side and the orbital side. Two
#: tolerances would eventually disagree, and the disagreement would surface
#: as a separation accepted from one direction and refused from the other.
EPOCH_MATCH_TOLERANCE_DAYS = INSTANT_MATCH_TOLERANCE_DAYS

REFERENCE_EPOCH_NOT_STATED = (
    "the astrometry has no stated reference epoch, so its position is not "
    "attached to any instant and cannot be moved to another one"
)
PROPER_MOTION_NOT_MEASURED = (
    "at least one proper-motion component is not published, and a missing "
    "component is not a measured zero, so the direction cannot be moved off "
    "the reference epoch"
)
RADIAL_VELOCITY_NOT_MEASURED = (
    "no radial velocity is published, so there is no rate of change along "
    "the line of sight and the distance cannot be moved off the reference "
    "epoch"
)
DISTANCE_NOT_USABLE = (
    "the astrometry has no usable distance, so there is a direction but no "
    "3D position to propagate"
)
DIRECTION_ONLY_NOT_A_POSITION = (
    "proper motion moves the sky direction only; combining it with a "
    "reference-epoch distance would report a cross-epoch 3D position that "
    "was never measured"
)

#: The model a direction-only propagation is produced under, stated on every
#: value it produces. This is not the same claim as
#: :data:`SPACE_MOTION_MODEL`: that one propagates a measured 3D velocity,
#: while this one has to supply a line-of-sight component the catalogue
#: never measured.
ZERO_RV_APPROXIMATION = (
    "radial velocity unavailable; the propagated direction uses the zero-RV, "
    "no-perspective-acceleration approximation - a first-order astrometric "
    "realization, not a uniquely determined space-motion solution"
)

DIRECTION_ONLY_IS_MODEL_DEPENDENT = (
    "the cross-epoch direction is model-dependent: with no radial velocity "
    "the perspective term is unconstrained, so physically possible "
    "line-of-sight motions give measurably different directions at this time"
)


class MotionKnowledge(str, Enum):
    """How much of a star's space motion the catalogue actually gives.

    Ordered by strength, and compared through :attr:`rank` rather than by
    enum identity so a caller can ask for "at least this much".
    """

    NONE = "NONE"
    """No reference epoch: the position is not attached to an instant."""

    REFERENCE_EPOCH_ONLY = "REFERENCE_EPOCH_ONLY"
    """A dated position. Complete at its own epoch and nowhere else."""

    DIRECTION_ONLY = "DIRECTION_ONLY"
    """Proper-motion information exists; complete 3D space motion does not.

    The name describes what the catalogue *has*, not what is known at
    another date. Cross-epoch direction in this tier is
    **model-dependent**: with no radial velocity the line-of-sight component
    of the space velocity is unconstrained, and a finite-distance star's
    apparent angular rate depends on it through perspective acceleration.
    What this tier propagates is therefore the zero-RV, no-perspective
    realization - see :data:`ZERO_RV_APPROXIMATION` - and never "the
    direction at t".
    """

    FULL_SPACE_MOTION = "FULL_SPACE_MOTION"
    """Proper motion and radial velocity: a full cross-epoch 3D state."""

    @property
    def rank(self) -> int:
        return {
            MotionKnowledge.NONE: 0,
            MotionKnowledge.REFERENCE_EPOCH_ONLY: 1,
            MotionKnowledge.DIRECTION_ONLY: 2,
            MotionKnowledge.FULL_SPACE_MOTION: 3,
        }[self]

    def __ge__(self, other) -> bool:  # type: ignore[override]
        if not isinstance(other, MotionKnowledge):
            return NotImplemented
        return self.rank >= other.rank

    @property
    def label(self) -> str:
        return {
            MotionKnowledge.NONE: "no reference epoch",
            MotionKnowledge.REFERENCE_EPOCH_ONLY: "position at a reference epoch only",
            MotionKnowledge.DIRECTION_ONLY: (
                "proper motion known, 3D space motion not; cross-epoch "
                "direction is a zero-RV approximation"
            ),
            MotionKnowledge.FULL_SPACE_MOTION: "full 3D space motion",
        }[self]


def _known(parameter: Parameter | None) -> bool:
    """True when a motion component was actually published.

    Assumptions do not count. A radial velocity substituted so a picture
    could be drawn must not open a gate that exists precisely to stop a
    substituted number from being propagated as though it were measured.
    """
    return parameter is not None and parameter.is_known and parameter.status.is_scientific


@dataclass(frozen=True)
class AstrometricState:
    """A star's astrometry as one catalogue published it, at one epoch.

    Every field is what the source said, converted but not reinterpreted.
    In particular :attr:`reference_epoch` is *stored*, never inferred: a
    position with no epoch column does not become J2000 because most
    catalogues use J2000, and Gaia's J2016.0 differs from J2000 by sixteen
    years of proper motion - 33 arcsec for HD 219134.

    ``pm_ra_cosdec`` is :math:`\\mu_\\alpha^* = \\dot\\alpha\\cos\\delta`,
    the great-circle rate, which is what Gaia's ``pmra`` already is and what
    Astropy's ``pm_ra_cosdec`` wants. Applying ``cos(dec)`` on the way in
    would square the factor and shrink every high-declination star's motion.
    """

    position: SkyPosition
    source_catalog: str = ""
    source_id: str | None = None
    release: str | None = None
    reference_epoch: Time | None = None
    pm_ra_cosdec: Parameter = unknown(u.mas / u.yr, provenance="pm_ra_cosdec")
    pm_dec: Parameter = unknown(u.mas / u.yr, provenance="pm_dec")
    radial_velocity: Parameter = unknown(u.km / u.s, provenance="radial_velocity")
    reference: str | None = None

    #: The *observed* state this one was propagated from, when this state is
    #: a model realization rather than a catalogue record.
    #:
    #: A realization must never be mistaken for an observation, and the
    #: specific way that could happen is chaining: propagate a direction-only
    #: star with an assumed zero radial velocity, read the differential
    #: Astropy produced, build a new state from it, and the missing
    #: measurement has quietly become a known one. Recording the origin lets
    #: :func:`propagate_astrometry` rebase onto it, so a second propagation
    #: starts from the observation rather than from the model output, and no
    #: chain of hops can compound an approximation into apparent knowledge.
    realized_from: "AstrometricState | None" = None

    @property
    def name(self) -> str:
        return self.position.name

    @property
    def is_realization(self) -> bool:
        """True when this state was propagated rather than published."""
        return self.realized_from is not None

    def observed_root(self) -> "AstrometricState":
        """The published state at the bottom of any chain of realizations.

        Propagation always starts here. For rectilinear space motion this is
        also numerically the better answer - one transform instead of two -
        but the reason it is mandatory is epistemic, not numerical.
        """
        state = self
        seen = 0
        while state.realized_from is not None and seen < 64:
            state = state.realized_from
            seen += 1
        return state

    @property
    def has_reference_epoch(self) -> bool:
        return self.reference_epoch is not None

    @property
    def has_proper_motion(self) -> bool:
        """True only when **both** components were published.

        One component and a silent zero for the other is a motion at the
        wrong position angle, which looks entirely like a motion.
        """
        return _known(self.pm_ra_cosdec) and _known(self.pm_dec)

    @property
    def has_radial_velocity(self) -> bool:
        return _known(self.radial_velocity)

    @property
    def knowledge(self) -> MotionKnowledge:
        """The strongest tier this state supports."""
        if not self.has_reference_epoch:
            return MotionKnowledge.NONE
        if not self.has_proper_motion:
            return MotionKnowledge.REFERENCE_EPOCH_ONLY
        if not self.has_radial_velocity or not self.position.has_distance:
            return MotionKnowledge.DIRECTION_ONLY
        return MotionKnowledge.FULL_SPACE_MOTION

    @property
    def reference_epoch_jyear(self) -> float | None:
        """The reference epoch as a Julian year, for display."""
        if self.reference_epoch is None:
            return None
        return float(self.reference_epoch.jyear)

    def describe(self) -> list[str]:
        """Lines that say what was measured and what was not."""
        source = self.source_catalog or "unknown catalogue"
        if self.source_id:
            source = "{0} {1}".format(source, self.source_id)
        epoch = self.reference_epoch_jyear
        return [
            "Astrometric source: {0}".format(source),
            "Reference epoch:    {0}".format(
                "not stated" if epoch is None else "J{0:.4f}".format(epoch)
            ),
            "pm_ra_cosdec:       {0}".format(self.pm_ra_cosdec.format()),
            "pm_dec:             {0}".format(self.pm_dec.format()),
            "Radial velocity:    {0}".format(self.radial_velocity.format()),
            "Motion knowledge:   {0}".format(self.knowledge.label),
        ]

    def blockers_for(self, tier: MotionKnowledge) -> list[str]:
        """Why this state does not reach ``tier``. Empty when it does."""
        reasons: list[str] = []
        if not self.has_reference_epoch:
            reasons.append(REFERENCE_EPOCH_NOT_STATED)
        if tier.rank >= MotionKnowledge.DIRECTION_ONLY.rank and not self.has_proper_motion:
            reasons.append(PROPER_MOTION_NOT_MEASURED)
        if tier.rank >= MotionKnowledge.FULL_SPACE_MOTION.rank:
            if not self.has_radial_velocity:
                reasons.append(RADIAL_VELOCITY_NOT_MEASURED)
            if not self.position.has_distance:
                reasons.append(DISTANCE_NOT_USABLE)
        return reasons

    # -- the Astropy boundary --------------------------------------------
    def skycoord(self, *, with_motion: bool = True) -> SkyCoord | None:
        """The state as a :class:`~astropy.coordinates.SkyCoord`.

        Only components that were actually published are attached. That is
        the rule the whole module turns on: Astropy fills a missing
        differential with zero when it propagates, so a coordinate must
        never be handed a *partial* motion and then read as a 3D state.
        With ``with_motion=False`` the result is the bare direction and
        distance at the reference epoch, which is what direction-only
        propagation needs.
        """
        ra = self.position.ra.value_in(u.deg)
        dec = self.position.dec.value_in(u.deg)
        if ra is None or dec is None:
            return None

        fields: dict = {
            "ra": ra * u.deg,
            "dec": dec * u.deg,
            "frame": "icrs",
        }
        if self.reference_epoch is not None:
            fields["obstime"] = self.reference_epoch

        if not with_motion:
            return SkyCoord(**fields)

        if self.has_proper_motion:
            fields["pm_ra_cosdec"] = self.pm_ra_cosdec.value_in(u.mas / u.yr) * u.mas / u.yr
            fields["pm_dec"] = self.pm_dec.value_in(u.mas / u.yr) * u.mas / u.yr
        if self.position.has_distance:
            fields["distance"] = self.position.distance.value_in(u.pc) * u.pc
        if self.has_radial_velocity and self.position.has_distance:
            # A radial velocity without a distance is a rate with nothing to
            # scale: Astropy needs the pair to build a 3D differential.
            fields["radial_velocity"] = self.radial_velocity.value_in(u.km / u.s) * u.km / u.s
        return SkyCoord(**fields)


@dataclass(frozen=True)
class PropagatedAstrometry:
    """One star's astrometry evaluated at one concrete instant.

    This is what replaces ``epoch_resolved: bool``. The difference is that a
    boolean asserted a gate was satisfied, while this carries the state that
    satisfies it - so a consumer that wants to add a planet offset has the
    propagated RA and Dec to rebuild the tangent basis from, and a consumer
    that wants two objects at a common time can check that they are at one.
    """

    position: SkyPosition
    obstime: Time
    status: Status
    source_state: AstrometricState
    blockers: tuple[str, ...] = ()
    knowledge: MotionKnowledge = MotionKnowledge.NONE
    motion_applied: bool = False

    #: The motion **at** ``obstime``, which is not the motion at the
    #: reference epoch. A star's apparent proper motion changes as it moves:
    #: the tangential velocity is fixed in space, so the angular rate scales
    #: with distance, and a receding star's radial velocity grows as the
    #: space velocity turns further along the line of sight. Reusing the
    #: catalogue rates to propagate onward from here would drop that
    #: perspective term - about 0.6 arcsec on a round trip for a fast nearby
    #: star, which is far larger than any of the uncertainties involved.
    pm_ra_cosdec: Parameter | None = None
    pm_dec: Parameter | None = None
    radial_velocity: Parameter | None = None

    #: True when the state handed in was itself a realization, so
    #: propagation was rebased onto the observation behind it. Recorded
    #: rather than inferred, because :attr:`source_state` is the observation
    #: by then and no longer remembers that a hop was skipped.
    rebased: bool = False

    @property
    def name(self) -> str:
        return self.source_state.name

    def as_state(self) -> AstrometricState:
        """This result re-expressed as astrometry at ``obstime``.

        So a propagated state can be propagated again - to another date, or
        back to where it came from - without the caller having to know that
        the motion changed on the way.

        Two rules keep this from manufacturing knowledge:

        * the returned state carries :attr:`AstrometricState.realized_from`,
          so :func:`propagate_astrometry` rebases onto the *observation*
          rather than propagating a model output a second time;
        * the radial velocity is whatever
          :func:`propagate_astrometry` decided it was entitled to be. For a
          ``DIRECTION_ONLY`` source that is the original **UNKNOWN**, never
          the zero-RV differential Astropy returned. Reading that
          differential back would turn "nobody measured this" into a
          measured number in one assignment, and the resulting state would
          then rank as :attr:`MotionKnowledge.FULL_SPACE_MOTION`.
        """
        return AstrometricState(
            position=self.position,
            source_catalog=self.source_state.source_catalog,
            source_id=self.source_state.source_id,
            release=self.source_state.release,
            reference_epoch=self.obstime,
            pm_ra_cosdec=(
                self.source_state.pm_ra_cosdec
                if self.pm_ra_cosdec is None
                else self.pm_ra_cosdec
            ),
            pm_dec=self.source_state.pm_dec if self.pm_dec is None else self.pm_dec,
            radial_velocity=(
                self.source_state.radial_velocity
                if self.radial_velocity is None
                else self.radial_velocity
            ),
            reference=self.source_state.reference,
            realized_from=self.source_state,
        )

    @property
    def is_publishable(self) -> bool:
        """True when this may be used as a real 3D position at ``obstime``.

        Requires no blockers, a scientific status and an actual distance.
        A direction-only propagation fails here by construction, which is
        the point: it is a genuine result and it is not a position.
        """
        return (
            not self.blockers
            and self.status.is_scientific
            and self.position.has_distance
        )

    @property
    def is_at_reference_epoch(self) -> bool:
        """True when no motion had to be invented because none was needed."""
        epoch = self.source_state.reference_epoch
        if epoch is None:
            return False
        return abs(float(self.obstime.jd - epoch.jd)) <= EPOCH_MATCH_TOLERANCE_DAYS

    @property
    def obstime_jd(self) -> float:
        return float(self.obstime.jd)

    def at_same_time_as(self, other: "PropagatedAstrometry | None") -> bool:
        """True when two states were evaluated at the same instant.

        The comparison is between Astropy times, so two states reached
        through different scales still compare as the same instant rather
        than as two Julian-day numbers that happen to differ by 69 seconds.
        """
        if other is None:
            return False
        return abs(float((self.obstime - other.obstime).to_value(u.day))) <= (
            EPOCH_MATCH_TOLERANCE_DAYS
        )

    def describe(self) -> list[str]:
        lines = list(self.source_state.describe())
        lines.append("Evaluated at:       JD {0:.6f}".format(self.obstime_jd))
        lines.append(
            "Motion applied:     {0}".format(
                "none needed - requested time is the reference epoch"
                if self.is_at_reference_epoch
                else (
                    SPACE_MOTION_MODEL
                    if self.knowledge is MotionKnowledge.FULL_SPACE_MOTION
                    else ZERO_RV_APPROXIMATION
                )
                if self.motion_applied
                else "none"
            )
        )
        if self.rebased:
            lines.append(
                "Rebased:            propagated from the observed state, not "
                "from the intermediate realization it was handed"
            )
        for reason in self.blockers:
            lines.append("Blocked:            {0}".format(reason))
        return lines


def _epochs_match(a: Time, b: Time) -> bool:
    return abs(float((a - b).to_value(u.day))) <= EPOCH_MATCH_TOLERANCE_DAYS


def _propagated_position(
    state: AstrometricState,
    coord: SkyCoord,
    *,
    keep_distance: bool,
) -> SkyPosition:
    """Rebuild a :class:`SkyPosition` from a propagated coordinate.

    ``keep_distance`` is False for direction-only propagation, where the
    distance deliberately becomes UNKNOWN rather than being carried across
    from the reference epoch. That is not information being thrown away: the
    reference-epoch distance is still on
    :attr:`PropagatedAstrometry.source_state`, correctly attached to the
    epoch it belongs to.
    """
    # ``keep_distance`` doubles as "this was a full space-motion solution":
    # the distance is only carried across when a radial velocity determined
    # how it changed. So it also selects which model the angles were
    # produced under, and the angles say so rather than both cases quoting
    # the sentence that belongs to a measured 3D velocity.
    note = SPACE_MOTION_MODEL if keep_distance else ZERO_RV_APPROXIMATION
    ra = derived(
        float(coord.ra.to_value(u.deg)),
        u.deg,
        provenance="apply_space_motion(ra)",
        note=note,
    )
    dec = derived(
        float(coord.dec.to_value(u.deg)),
        u.deg,
        provenance="apply_space_motion(dec)",
        note=note,
    )
    if keep_distance and coord.data.distance is not None:
        distance = derived(
            float(coord.distance.to_value(u.pc)),
            u.pc,
            provenance="apply_space_motion(distance)",
            note=SPACE_MOTION_MODEL,
        )
    else:
        distance = unknown(
            u.pc,
            provenance="apply_space_motion",
            note=DIRECTION_ONLY_NOT_A_POSITION,
        )
    return SkyPosition(
        name=state.position.name,
        ra=ra,
        dec=dec,
        distance=distance,
        frame=Frame.ICRS,
    )


def _propagated_motion(
    state: AstrometricState, coord: SkyCoord, *, full: bool
) -> dict[str, Parameter]:
    """The motion the propagated coordinate now carries.

    ERFA's ``starpm`` returns the velocity at the new epoch, not the one it
    was given, and the difference is the perspective term: a star's angular
    rate scales inversely with its distance, and its radial velocity picks
    up whatever component of the space velocity has turned along the new
    line of sight. Discarding that and re-using the catalogue rates is a
    quiet way to make a second propagation wrong.
    """
    # A direction-only propagation had to supply the line-of-sight component
    # the catalogue never measured, so every value it produces says which
    # model produced it rather than borrowing the space-motion sentence.
    note = SPACE_MOTION_MODEL if full else ZERO_RV_APPROXIMATION
    motion: dict[str, Parameter] = {
        "pm_ra_cosdec": derived(
            float(coord.pm_ra_cosdec.to_value(u.mas / u.yr)),
            u.mas / u.yr,
            provenance="apply_space_motion(pm_ra_cosdec)",
            note=note,
        ),
        "pm_dec": derived(
            float(coord.pm_dec.to_value(u.mas / u.yr)),
            u.mas / u.yr,
            provenance="apply_space_motion(pm_dec)",
            note=note,
        ),
    }
    if full:
        motion["radial_velocity"] = derived(
            float(coord.radial_velocity.to_value(u.km / u.s)),
            u.km / u.s,
            provenance="apply_space_motion(radial_velocity)",
            note=SPACE_MOTION_MODEL,
        )
    else:
        # No radial velocity went in, so none comes out. Reading a zero off
        # a unit-spherical coordinate would manufacture the measurement the
        # direction-only tier exists to say is absent.
        motion["radial_velocity"] = state.radial_velocity
    return motion


def _reference_status(state: AstrometricState) -> Status:
    """The status a position inherits from the angles and distance behind it."""
    parts = (state.position.ra, state.position.dec, state.position.distance)
    if any(not p.is_known for p in parts):
        return Status.UNKNOWN
    if any(p.status is Status.ASSUMED_FOR_VISUALIZATION for p in parts):
        return Status.ASSUMED_FOR_VISUALIZATION
    return Status.MEASURED if all(p.status is Status.MEASURED for p in parts) else Status.DERIVED


def propagate_astrometry(
    state: AstrometricState | None,
    obstime: Time,
    *,
    require: MotionKnowledge = MotionKnowledge.FULL_SPACE_MOTION,
) -> PropagatedAstrometry | None:
    """Evaluate ``state`` at ``obstime``, reporting what it could not do.

    Returns ``None`` only when there is no state at all; every other outcome
    is a :class:`PropagatedAstrometry` that either is publishable or says
    why it is not. A failure is a result here rather than an exception,
    because "this host's radial velocity was never measured" is a permanent
    property of the catalogue and not an error condition.

    ``require`` is the tier the caller needs. It defaults to
    :attr:`MotionKnowledge.FULL_SPACE_MOTION` because that is what adding an
    AU-scale offset to a parsec-scale position at an arbitrary date demands;
    a caller that only wants a direction can ask for less and get a result
    that says so.

    The one case with no blockers and no motion is the important one: when
    ``obstime`` *is* the reference epoch, nothing has to be invented, and a
    star with a dated position and no measured motion at all is fully
    publishable there. Demanding proper motion in order to stand still would
    withhold the one answer that needs no model.

    A state that is itself a realization is **rebased onto the observation
    it came from** before anything is computed. Propagating a model output
    would compound its assumption: a direction-only star realized at
    zero radial velocity, then realized again from that, has had the
    approximation applied twice while looking exactly like one careful
    propagation. Rectilinear motion makes the rebased answer identical for a
    full solution, so nothing is lost by doing it unconditionally.
    """
    if state is None:
        return None

    rebased = state.is_realization
    state = state.observed_root()

    if not state.has_reference_epoch:
        return PropagatedAstrometry(
            position=state.position,
            obstime=obstime,
            status=Status.UNKNOWN,
            source_state=state,
            rebased=rebased,
            blockers=(REFERENCE_EPOCH_NOT_STATED,),
            knowledge=MotionKnowledge.NONE,
        )

    epoch = state.reference_epoch
    assert epoch is not None  # narrowed by has_reference_epoch

    # -- the requested time is the reference epoch -----------------------
    if _epochs_match(obstime, epoch):
        blockers: tuple[str, ...] = ()
        if require.rank >= MotionKnowledge.FULL_SPACE_MOTION.rank and not (
            state.position.has_distance
        ):
            blockers = (DISTANCE_NOT_USABLE,)
        return PropagatedAstrometry(
            position=state.position,
            obstime=obstime,
            status=_reference_status(state),
            source_state=state,
            rebased=rebased,
            blockers=blockers,
            knowledge=state.knowledge,
            motion_applied=False,
        )

    # -- a different instant: motion is required -------------------------
    reasons = state.blockers_for(require)
    coord = state.skycoord()
    if coord is None:
        return PropagatedAstrometry(
            position=state.position,
            obstime=obstime,
            status=Status.UNKNOWN,
            source_state=state,
            rebased=rebased,
            blockers=tuple(reasons) or (REFERENCE_EPOCH_NOT_STATED,),
            knowledge=state.knowledge,
        )

    tier = state.knowledge
    if tier.rank < MotionKnowledge.DIRECTION_ONLY.rank:
        # Nothing to propagate with. The reference-epoch position is
        # returned unchanged and blocked, rather than being silently
        # advertised as the position at the requested date.
        return PropagatedAstrometry(
            position=state.position,
            obstime=obstime,
            status=Status.UNKNOWN,
            source_state=state,
            rebased=rebased,
            blockers=tuple(reasons) or (PROPER_MOTION_NOT_MEASURED,),
            knowledge=tier,
        )

    full = tier is MotionKnowledge.FULL_SPACE_MOTION
    if not full:
        # Direction-only: strip the distance so the coordinate carries a
        # purely tangential motion. Leaving the distance on with no radial
        # velocity would let Astropy propagate a line-of-sight rate of zero
        # that nobody measured.
        coord = state.skycoord(with_motion=False)
        assert coord is not None
        coord = SkyCoord(
            ra=coord.ra,
            dec=coord.dec,
            pm_ra_cosdec=state.pm_ra_cosdec.value_in(u.mas / u.yr) * u.mas / u.yr,
            pm_dec=state.pm_dec.value_in(u.mas / u.yr) * u.mas / u.yr,
            obstime=epoch,
            frame="icrs",
        )
        for reason in (
            DIRECTION_ONLY_NOT_A_POSITION,
            DIRECTION_ONLY_IS_MODEL_DEPENDENT,
        ):
            if reason not in reasons:
                reasons.append(reason)

    if full:
        moved = coord.apply_space_motion(new_obstime=obstime)
    else:
        # A direction-only coordinate carries no parallax, so ERFA's
        # ``pmsafe`` reports that it substituted a nominal distance for the
        # light-time term it cannot evaluate. That is precisely the
        # situation being modelled - there is no distance - and the
        # tangential propagation it returns is the answer wanted, so the
        # warning is silenced here, narrowly, rather than left to surface as
        # noise that every consumer learns to ignore.
        import warnings

        from erfa import ErfaWarning

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=ErfaWarning,
                message=".*distance overridden.*",
            )
            moved = coord.apply_space_motion(new_obstime=obstime)

    position = _propagated_position(state, moved, keep_distance=full)
    motion = _propagated_motion(state, moved, full=full)

    status = Status.UNKNOWN if reasons else Status.DERIVED
    return PropagatedAstrometry(
        position=position,
        obstime=obstime,
        status=status,
        source_state=state,
        blockers=tuple(reasons),
        knowledge=tier,
        motion_applied=True,
        rebased=rebased,
        **motion,
    )


def at_reference_epoch(state: AstrometricState | None) -> PropagatedAstrometry | None:
    """The state at its own reference epoch - the no-model answer.

    A convenience for the tier the knowledge table calls "reference-epoch 3D
    position": if the requested time equals the reference epoch, no motion
    needs to be invented. Returns None when there is no epoch to evaluate
    at, because there is then no instant this could be an answer for.
    """
    if state is None or state.reference_epoch is None:
        return None
    return propagate_astrometry(state, state.reference_epoch)


def astrometric_state_from_position(
    position: SkyPosition,
    *,
    source_catalog: str = "",
    reference_epoch: Time | None = None,
) -> AstrometricState:
    """Wrap a bare :class:`SkyPosition` with no motion and no epoch.

    Exists so a caller holding only the NASA archive's RA/Dec can enter the
    astrometric pipeline at all - and immediately be told, by the blockers
    on the result, that a position with no stated epoch cannot be moved.
    That is a better shape than an optional astrometry argument that is
    ``None`` half the time, because the reason travels.
    """
    return AstrometricState(
        position=position,
        source_catalog=source_catalog,
        reference_epoch=reference_epoch,
    )


__all__ += ["astrometric_state_from_position"]
