"""Scientific constants sourced from Astropy (roadmap section 3.8).

The original prototype hard-coded CODATA numbers in the GUI module.  Every
constant used by the scientific core now comes from
:mod:`astropy.constants`, so values are traceable and carry units.

Nothing in this module may define a physical constant by hand.
"""

from __future__ import annotations

import astropy.units as u
from astropy.constants import G, L_sun, M_earth, M_jup, M_sun, R_earth, R_jup, R_sun
from astropy.constants import au as AU
from astropy.constants import c as C_LIGHT
from astropy.constants import h as H_PLANCK
from astropy.constants import k_B as K_BOLTZMANN
from astropy.constants import pc as PARSEC
from astropy.constants import sigma_sb as SIGMA_SB

__all__ = [
    "G",
    "C_LIGHT",
    "H_PLANCK",
    "K_BOLTZMANN",
    "SIGMA_SB",
    "AU",
    "PARSEC",
    "M_SUN",
    "R_SUN",
    "L_SUN",
    "M_JUP",
    "R_JUP",
    "M_EARTH",
    "R_EARTH",
    "B_WIEN",
    "SOLAR_EFFECTIVE_TEMPERATURE",
    "AU_PER_PARSEC",
    "PARSEC_PER_AU",
    "EARTH_TO_SOLAR_MASS",
    "JUPITER_TO_SOLAR_MASS",
]

M_SUN = M_sun
R_SUN = R_sun
L_SUN = L_sun
M_JUP = M_jup
R_JUP = R_jup
M_EARTH = M_earth
R_EARTH = R_earth

#: Wien displacement constant, derived from h, c and k_B rather than quoted.
#: b = h*c / (k_B * x) with x the root of x = 5*(1 - exp(-x)).
_WIEN_X = 4.965114231744276
B_WIEN = (H_PLANCK * C_LIGHT / (K_BOLTZMANN * _WIEN_X)).to(u.m * u.K)

#: IAU 2015 nominal solar effective temperature.
SOLAR_EFFECTIVE_TEMPERATURE = 5772.0 * u.K

#: Exact unit bridges.  Roadmap section 4.2: 1 AU is 4.8481368e-6 pc, not 0.005.
PARSEC_PER_AU = float((1.0 * u.au).to_value(u.pc))
AU_PER_PARSEC = float((1.0 * u.pc).to_value(u.au))

EARTH_TO_SOLAR_MASS = float((M_EARTH / M_SUN).decompose().value)
JUPITER_TO_SOLAR_MASS = float((M_JUP / M_SUN).decompose().value)
