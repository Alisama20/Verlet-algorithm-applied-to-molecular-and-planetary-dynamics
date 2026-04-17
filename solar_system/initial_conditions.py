"""Initial conditions for the 8-planet Solar System.

Rescaled units (heliocentric, G M_sun = 1):

    [length]  = 1 AU
    [mass]    = 1 M_sun
    [time]    = (AU^3 / G M_sun)^{1/2} ~ 58.1 days  (~= 1 yr / 2 pi)

so Kepler's third law takes the form  T^2 = (2 pi)^2 * a^3.
"""

import numpy as np

# One time unit in days (the C++ code uses 58.1).
TIME_UNIT_DAYS = 58.1

# Real orbital periods in days, for validation.
REAL_PERIODS_DAYS = np.array([
    0.0,        # Sun (placeholder)
    88.0,       # Mercury
    224.7,      # Venus
    365.2,      # Earth
    687.0,      # Mars
    4331.0,     # Jupiter
    10747.0,    # Saturn
    30589.0,    # Uranus
    59800.0,    # Neptune
])

PLANET_NAMES = [
    "Sun",
    "Mercury",
    "Venus",
    "Earth",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
]


def solar_system_ic():
    """Return (mass, pos, vel) arrays for the Sun + 8 planets.

    ``pos`` and ``vel`` have shape (9, 2): x, y components.  All planets are
    placed on the +x axis with velocity purely in +y, giving counter-clockwise
    orbits.  Velocities come from the same rescaled values used in the original
    C++ solver so the two match bit-for-bit in the first few steps.
    """
    mass = np.array([
        1.0,                # Sun
        1.65158371e-7,      # Mercury
        2.435e-6,           # Venus
        3.002513825e-6,     # Earth
        3.21e-7,            # Mars
        9.49e-4,            # Jupiter
        2.84e-4,            # Saturn
        4.34e-5,            # Uranus
        5.1e-5,             # Neptune
    ])

    x = np.array([0.0, 0.386, 0.71233, 1.0, 1.52, 5.19, 9.55, 19.1133, 30.1])
    y = np.zeros_like(x)

    vx = np.zeros_like(x)
    vy = np.array([
        0.0,
        1.58944702,
        1.173642314,
        1.005979127,
        0.8081,
        0.439278,
        0.325266,
        0.2280,
        0.18107,
    ])

    pos = np.column_stack([x, y])
    vel = np.column_stack([vx, vy])
    return mass, pos, vel
