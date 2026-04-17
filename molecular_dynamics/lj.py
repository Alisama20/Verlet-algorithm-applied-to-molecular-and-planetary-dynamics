"""2D Lennard-Jones fluid with periodic boundary conditions.

Reduced (LJ) units: epsilon = sigma = m = k_B = 1.  In these units the
Lennard-Jones pair potential is

    V(r) = 4 [ (1 / r)^12 - (1 / r)^6 ]

shifted and truncated smoothly at r = r_cut (default 2.5 sigma).  The force
on particle i is

    F_i = - sum_{j != i}  dV/dr * (r_i - r_j) / r

with the usual minimum-image convention inside a square box of side L.

The ``simulate`` routine integrates the equations of motion with
Velocity-Verlet and records potential / kinetic / total energy and
instantaneous temperature T = <v^2> / d (d = 2 in 2D, ignoring the factor of
2/d that would give the thermodynamic temperature; see ``kT_inst``).

All hot loops are Numba-jitted with ``parallel=True``.
"""

import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------------
# Force evaluation — O(N^2) but parallel over particles
# ---------------------------------------------------------------------------

@njit(parallel=True, fastmath=True, cache=True)
def _forces_and_potential(pos, L, r_cut, force, shift):
    """Compute LJ forces and total potential energy.

    Parameters
    ----------
    pos : (N, 2) array   particle positions (inside [0, L)^2)
    L   : float          box side length
    r_cut : float        cutoff radius
    force : (N, 2) array (output) force on each particle
    shift : float        potential shift so V(r_cut) = 0

    Returns
    -------
    V_tot : float        total potential energy of the system
    """
    N = pos.shape[0]
    r_cut2 = r_cut * r_cut

    # Each thread accumulates its own potential — numba reduces prange sums.
    V_thread = np.zeros(N)

    for i in prange(N):
        fxi = 0.0
        fyi = 0.0
        xi = pos[i, 0]
        yi = pos[i, 1]
        for j in range(N):
            if i == j:
                continue
            dx = xi - pos[j, 0]
            dy = yi - pos[j, 1]
            # Minimum image convention
            dx -= L * np.round(dx / L)
            dy -= L * np.round(dy / L)
            r2 = dx * dx + dy * dy
            if r2 >= r_cut2 or r2 < 1e-12:
                continue
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6
            # F = 24 * (2 / r^13 - 1 / r^7) * r_hat
            f_over_r = 24.0 * (2.0 * inv_r12 - inv_r6) * inv_r2
            fxi += f_over_r * dx
            fyi += f_over_r * dy
            if j > i:
                V_thread[i] += 4.0 * (inv_r12 - inv_r6) - shift
        force[i, 0] = fxi
        force[i, 1] = fyi

    V_tot = 0.0
    for i in range(N):
        V_tot += V_thread[i]
    return V_tot


@njit(fastmath=True, cache=True)
def _wrap(pos, L):
    """Apply periodic boundary conditions."""
    N = pos.shape[0]
    for i in range(N):
        if pos[i, 0] < 0.0:
            pos[i, 0] += L
        elif pos[i, 0] >= L:
            pos[i, 0] -= L
        if pos[i, 1] < 0.0:
            pos[i, 1] += L
        elif pos[i, 1] >= L:
            pos[i, 1] -= L


@njit(fastmath=True, cache=True)
def _kinetic(vel):
    N = vel.shape[0]
    s = 0.0
    for i in range(N):
        s += 0.5 * (vel[i, 0] ** 2 + vel[i, 1] ** 2)
    return s


# ---------------------------------------------------------------------------
# Velocity-Verlet driver
# ---------------------------------------------------------------------------

@njit(fastmath=True, cache=True)
def _run(pos0, vel0, L, r_cut, shift, dt, n_steps, save_every):
    N = pos0.shape[0]
    pos = pos0.copy()
    vel = vel0.copy()
    force = np.zeros_like(pos)
    force_new = np.zeros_like(pos)

    n_frames = n_steps // save_every + 1
    traj = np.empty((n_frames, N, 2))
    KE = np.empty(n_frames)
    PE = np.empty(n_frames)
    T_arr = np.empty(n_frames)
    times = np.empty(n_frames)

    V = _forces_and_potential(pos, L, r_cut, force, shift)
    traj[0] = pos
    PE[0] = V
    KE[0] = _kinetic(vel)
    T_arr[0] = KE[0] / N  # 2D: <v^2/2> per particle == kT
    times[0] = 0.0

    frame = 1
    for k in range(1, n_steps + 1):
        # r(t + dt) = r(t) + v(t) dt + F(t) dt^2 / 2  (m = 1)
        for i in range(N):
            pos[i, 0] += vel[i, 0] * dt + 0.5 * force[i, 0] * dt * dt
            pos[i, 1] += vel[i, 1] * dt + 0.5 * force[i, 1] * dt * dt
        _wrap(pos, L)

        V = _forces_and_potential(pos, L, r_cut, force_new, shift)

        for i in range(N):
            vel[i, 0] += 0.5 * (force[i, 0] + force_new[i, 0]) * dt
            vel[i, 1] += 0.5 * (force[i, 1] + force_new[i, 1]) * dt

        for i in range(N):
            force[i, 0] = force_new[i, 0]
            force[i, 1] = force_new[i, 1]

        if k % save_every == 0:
            traj[frame] = pos
            PE[frame] = V
            KE[frame] = _kinetic(vel)
            T_arr[frame] = KE[frame] / N
            times[frame] = k * dt
            frame += 1

    return traj[:frame], KE[:frame], PE[:frame], T_arr[:frame], times[:frame]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def lj_initial_conditions(n_side=10, rho=0.6, T=1.0, seed=0):
    """Square lattice of ``n_side`` x ``n_side`` particles at reduced density
    ``rho`` and Maxwell-Boltzmann velocities drawn at temperature ``T``.

    Returns (pos, vel, L).
    """
    N = n_side * n_side
    L = np.sqrt(N / rho)
    a = L / n_side

    xs = (np.arange(n_side) + 0.5) * a
    ys = (np.arange(n_side) + 0.5) * a
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pos = np.column_stack([X.ravel(), Y.ravel()])

    rng = np.random.default_rng(seed)
    vel = rng.standard_normal((N, 2)) * np.sqrt(T)
    vel -= vel.mean(axis=0)  # kill centre-of-mass drift

    # Rescale to exactly the requested temperature (2D: kT = <v^2>/2).
    T_now = 0.5 * np.sum(vel * vel) / N
    vel *= np.sqrt(T / T_now)

    return pos, vel, float(L)


def simulate(pos, vel, L, dt=0.005, n_steps=5_000, save_every=10, r_cut=2.5):
    """Integrate a 2D Lennard-Jones fluid with Velocity-Verlet.

    Returns (traj, KE, PE, T, times).
    """
    shift = 4.0 * ((1.0 / r_cut) ** 12 - (1.0 / r_cut) ** 6)
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    vel = np.ascontiguousarray(vel, dtype=np.float64)
    return _run(pos, vel, float(L), float(r_cut), float(shift),
                float(dt), int(n_steps), int(save_every))
