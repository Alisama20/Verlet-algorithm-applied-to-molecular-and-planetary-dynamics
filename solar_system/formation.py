"""Planet-formation toy model — N planetesimals orbit the Sun and merge on
contact.

Kept deliberately simple (only Sun-planetesimal gravity, perfectly inelastic
collisions, 2D) so the statistics of the final body distribution can be
inspected visually after 1000 steps from 1000 initial planetesimals.
"""

import numpy as np
from numba import njit, prange

# Same units as :mod:`solar_system.initial_conditions`: G M_sun = 1.
# Average density of the protosolar disk, in M_sun / AU^3.
DENSIDAD_MEDIA = 2.11e-61


@njit(cache=True)
def _step_sun_only(pos, vel, acc, dt):
    """Velocity-Verlet step under the Sun's gravity only (Sun at origin)."""
    N = pos.shape[0]

    # Drift + kick-1: positions forward, cache old acceleration.
    for i in range(N):
        pos[i, 0] += vel[i, 0] * dt + 0.5 * acc[i, 0] * dt * dt
        pos[i, 1] += vel[i, 1] * dt + 0.5 * acc[i, 1] * dt * dt

    ax_old = acc[:, 0].copy()
    ay_old = acc[:, 1].copy()

    for i in range(N):
        x = pos[i, 0]
        y = pos[i, 1]
        r2 = x * x + y * y
        inv_r3 = 1.0 / (r2 * np.sqrt(r2))
        acc[i, 0] = -x * inv_r3
        acc[i, 1] = -y * inv_r3

    for i in range(N):
        vel[i, 0] += 0.5 * (ax_old[i] + acc[i, 0]) * dt
        vel[i, 1] += 0.5 * (ay_old[i] + acc[i, 1]) * dt


@njit(parallel=True, cache=True)
def _collide_inplace(pos, vel, mass, radius, alive):
    """Detect pairwise collisions and merge perfectly inelastically.

    ``alive[i]`` is flipped to 0 when body i is absorbed into another.  The
    loop over pairs is O(N^2) but parallel over the outer index.
    """
    N = pos.shape[0]
    collisions = 0
    for i in prange(N):
        if alive[i] == 0:
            continue
        xi, yi = pos[i, 0], pos[i, 1]
        ri = radius[i]
        mi = mass[i]
        for j in range(i + 1, N):
            if alive[j] == 0:
                continue
            dx = pos[j, 0] - xi
            dy = pos[j, 1] - yi
            d2 = dx * dx + dy * dy
            rsum = ri + radius[j]
            if d2 <= rsum * rsum:
                mj = mass[j]
                mtot = mi + mj
                # Momentum-conserving merge, size keeps the density constant.
                vel[i, 0] = (mi * vel[i, 0] + mj * vel[j, 0]) / mtot
                vel[i, 1] = (mi * vel[i, 1] + mj * vel[j, 1]) / mtot
                pos[i, 0] = (mi * xi + mj * pos[j, 0]) / mtot
                pos[i, 1] = (mi * yi + mj * pos[j, 1]) / mtot
                radius[i] = (ri ** 3 + radius[j] ** 3) ** (1.0 / 3.0)
                mass[i] = mtot
                mi = mtot
                alive[j] = 0
                collisions += 1
    return collisions


def _initial_planetesimals(n, rng, r_min=0.387, r_max=39.440):
    """Populate a protoplanetary disk with ``n`` planetesimals."""
    # All bodies start with the same radius (1000 km) and density.
    radius = np.full(n, 1000.0 / 1.496e8)  # AU
    mass = np.full(n, DENSIDAD_MEDIA * 4.0 * np.pi * radius[0] ** 3 / 3.0)

    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    dist = rng.uniform(r_min, r_max, n)

    pos = np.column_stack([dist * np.cos(theta), dist * np.sin(theta)])

    # Tangential velocity with some spread so orbits eventually cross.
    jitter = rng.uniform(-np.pi / 5.0, np.pi / 5.0, n)
    v_ang = theta + jitter
    v_mod = 1.0 / np.sqrt(dist)
    vel = np.column_stack([-v_mod * np.sin(v_ang), v_mod * np.cos(v_ang)])

    return mass, radius, pos, vel


def simulate_formation(
    n_planetesimals=1000,
    n_steps=1000,
    dt=1.0,
    growth_factor=50.0,
    seed=2026,
):
    """Run the planetesimal-formation toy model.

    The ``growth_factor`` multiplies the physical radius used for collisions
    so the simulation produces a visible number of merges in 1000 steps.

    Returns (initial_pos, final_pos, final_mass, collisions).
    """
    rng = np.random.default_rng(seed)
    mass, radius, pos, vel = _initial_planetesimals(n_planetesimals, rng)
    radius *= growth_factor  # enlarge collision cross-sections

    initial_pos = pos.copy()

    acc = np.zeros_like(pos)
    for i in range(n_planetesimals):
        x, y = pos[i]
        r2 = x * x + y * y
        inv_r3 = 1.0 / (r2 * np.sqrt(r2))
        acc[i, 0] = -x * inv_r3
        acc[i, 1] = -y * inv_r3

    alive = np.ones(n_planetesimals, dtype=np.int64)
    total_collisions = 0
    for _ in range(n_steps):
        _step_sun_only(pos, vel, acc, dt)
        total_collisions += _collide_inplace(pos, vel, mass, radius, alive)

    survivors = alive.astype(bool)
    return initial_pos, pos[survivors], mass[survivors], total_collisions
