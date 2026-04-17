"""Velocity-Verlet N-body integrator, Numba-accelerated and parallel.

Units are heliocentric with G M_sun = 1 (see
``solar_system.initial_conditions``).  The integrator stores the full
trajectory so it can be rendered later.

Complexity is O(N^2) per step; for N = 9 bodies that is irrelevant, but the
Lennard-Jones simulation in :mod:`molecular_dynamics` re-uses the same
parallel pattern for much larger systems.
"""

import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------------
# Force evaluation
# ---------------------------------------------------------------------------

@njit(parallel=True, fastmath=True, cache=True)
def _accelerations(mass, pos, acc):
    """Fill ``acc`` with the gravitational acceleration on every body.

    The pairwise loop is parallelised over the outer index ``i``; reads of
    ``mass[j]`` and ``pos[j]`` are shared across threads without conflict.
    """
    N = mass.shape[0]
    for i in prange(N):
        ax = 0.0
        ay = 0.0
        xi = pos[i, 0]
        yi = pos[i, 1]
        for j in range(N):
            if i == j:
                continue
            dx = pos[j, 0] - xi
            dy = pos[j, 1] - yi
            r2 = dx * dx + dy * dy
            inv_r3 = 1.0 / (r2 * np.sqrt(r2))
            ax += mass[j] * dx * inv_r3
            ay += mass[j] * dy * inv_r3
        acc[i, 0] = ax
        acc[i, 1] = ay


@njit(parallel=True, fastmath=True, cache=True)
def _total_energy(mass, pos, vel):
    """Kinetic + gravitational potential energy of the whole system."""
    N = mass.shape[0]

    ke = 0.0
    for i in prange(N):
        ke += 0.5 * mass[i] * (vel[i, 0] ** 2 + vel[i, 1] ** 2)

    pe = 0.0
    for i in prange(N):
        s = 0.0
        for j in range(i + 1, N):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            s -= mass[i] * mass[j] / np.sqrt(dx * dx + dy * dy)
        pe += s

    return ke + pe


# ---------------------------------------------------------------------------
# Velocity-Verlet stepper
# ---------------------------------------------------------------------------

@njit(fastmath=True, cache=True)
def _run(mass, pos0, vel0, dt, n_steps, save_every):
    """Core loop — does not allocate objects on the hot path."""
    N = mass.shape[0]
    pos = pos0.copy()
    vel = vel0.copy()
    acc = np.zeros_like(pos)
    acc_new = np.zeros_like(pos)

    n_frames = n_steps // save_every + 1
    traj = np.empty((n_frames, N, 2))
    energies = np.empty(n_frames)
    times = np.empty(n_frames)

    _accelerations(mass, pos, acc)

    traj[0] = pos
    energies[0] = _total_energy(mass, pos, vel)
    times[0] = 0.0

    frame = 1
    for k in range(1, n_steps + 1):
        # r(t + dt) = r(t) + v(t) dt + a(t) dt^2 / 2
        for i in range(N):
            pos[i, 0] += vel[i, 0] * dt + 0.5 * acc[i, 0] * dt * dt
            pos[i, 1] += vel[i, 1] * dt + 0.5 * acc[i, 1] * dt * dt

        _accelerations(mass, pos, acc_new)

        # v(t + dt) = v(t) + [a(t) + a(t + dt)] dt / 2
        for i in range(N):
            vel[i, 0] += 0.5 * (acc[i, 0] + acc_new[i, 0]) * dt
            vel[i, 1] += 0.5 * (acc[i, 1] + acc_new[i, 1]) * dt

        for i in range(N):
            acc[i, 0] = acc_new[i, 0]
            acc[i, 1] = acc_new[i, 1]

        if k % save_every == 0:
            traj[frame] = pos
            energies[frame] = _total_energy(mass, pos, vel)
            times[frame] = k * dt
            frame += 1

    return traj[:frame], energies[:frame], times[:frame]


def simulate(mass, pos0, vel0, dt=1e-3, n_steps=20_000, save_every=10):
    """Integrate the N-body system with Velocity-Verlet.

    Parameters
    ----------
    mass, pos0, vel0 : (N,), (N, 2), (N, 2) arrays
        Masses and initial phase-space coordinates.
    dt : float
        Time step in the natural units (~58.1 days per unit).
    n_steps : int
        Number of integration steps.
    save_every : int
        Record a frame every ``save_every`` steps.

    Returns
    -------
    traj : (F, N, 2) array
        Positions of every body at every saved frame.
    energies : (F,) array
        Total mechanical energy at every saved frame.
    times : (F,) array
        Simulation time (natural units) at every saved frame.
    """
    mass = np.ascontiguousarray(mass, dtype=np.float64)
    pos0 = np.ascontiguousarray(pos0, dtype=np.float64)
    vel0 = np.ascontiguousarray(vel0, dtype=np.float64)
    return _run(mass, pos0, vel0, float(dt), int(n_steps), int(save_every))


# ---------------------------------------------------------------------------
# Period measurement (independent of the integrator)
# ---------------------------------------------------------------------------

def measure_periods(traj, times, time_unit_days=58.1):
    """Measure one orbital period per body by detecting the first return to
    the initial angular position after at least a quarter of a turn.

    Returns an array of periods **in days**, with 0.0 for bodies at the origin.
    """
    F, N, _ = traj.shape
    periods = np.zeros(N)
    for i in range(N):
        x0, y0 = traj[0, i]
        r0 = np.hypot(x0, y0)
        if r0 < 1e-12:
            continue
        theta0 = np.arctan2(y0, x0)

        # Unwrap the angle along the trajectory.
        theta = np.arctan2(traj[:, i, 1], traj[:, i, 0])
        theta = np.unwrap(theta - theta0)  # zero at t = 0

        sign = np.sign(theta[10] - theta[0]) or 1.0
        target = sign * 2.0 * np.pi

        # First frame where |theta| crossed 2 pi.
        if sign > 0:
            idx = np.where(theta >= target)[0]
        else:
            idx = np.where(theta <= target)[0]

        if idx.size == 0:
            periods[i] = np.nan
        else:
            # Linear interpolation between idx-1 and idx.
            k = idx[0]
            if k == 0:
                periods[i] = times[0]
            else:
                t1, t2 = times[k - 1], times[k]
                a1, a2 = theta[k - 1], theta[k]
                frac = (target - a1) / (a2 - a1)
                periods[i] = (t1 + frac * (t2 - t1)) * time_unit_days
    return periods
