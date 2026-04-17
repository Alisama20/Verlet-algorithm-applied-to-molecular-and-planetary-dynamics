"""Run a 2D Lennard-Jones simulation and save four figures.

Outputs (all under ``figures/``):

    md_energy.png      — kinetic, potential and total energy vs time
    md_temperature.png — instantaneous temperature vs time
    md_snapshots.png   — particle positions at t = 0 and t = t_end
    md_rdf.png         — radial distribution function g(r) at steady state
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from molecular_dynamics import simulate, lj_initial_conditions

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(FIGDIR, exist_ok=True)

# Simulation parameters
N_SIDE = 10     # 10×10 = 100 particles
RHO = 0.6       # reduced number density
T_INIT = 1.0    # initial temperature (LJ units)
DT = 0.005
N_STEPS = 5_000
SAVE_EVERY = 10
R_CUT = 2.5


def _rdf(pos, L, n_bins=100, r_max=None):
    """Radial distribution function g(r) via histogram of pair distances."""
    if r_max is None:
        r_max = 0.5 * L
    N = pos.shape[0]
    rho = N / L ** 2
    dr = r_max / n_bins
    counts = np.zeros(n_bins)
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dx -= L * np.round(dx / L)
            dy -= L * np.round(dy / L)
            r = np.hypot(dx, dy)
            if r < r_max:
                k = int(r / dr)
                if k < n_bins:
                    counts[k] += 2  # pair counted once, contributes to both
    r_edges = np.linspace(0.0, r_max, n_bins + 1)
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    # Normalise by ideal-gas shell area and density
    shell_area = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
    gr = counts / (N * rho * shell_area)
    return r_mid, gr


def main():
    pos0, vel0, L = lj_initial_conditions(n_side=N_SIDE, rho=RHO, T=T_INIT, seed=0)
    N = pos0.shape[0]

    print(f"Running LJ simulation: N = {N}, rho = {RHO}, T = {T_INIT}, "
          f"dt = {DT}, steps = {N_STEPS} ...")
    traj, KE, PE, T_arr, times = simulate(
        pos0, vel0, L, dt=DT, n_steps=N_STEPS, save_every=SAVE_EVERY, r_cut=R_CUT
    )
    print(f"  {traj.shape[0]} frames recorded")
    E_tot = KE + PE

    # ---- energy vs time -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, KE, label="Kinetic", lw=0.8)
    ax.plot(times, PE, label="Potential", lw=0.8)
    ax.plot(times, E_tot, "k-", label="Total", lw=1.2)
    ax.set_xlabel(r"time  [$\tau_{LJ}$]")
    ax.set_ylabel(r"energy  [$\varepsilon$]")
    ax.set_title(f"LJ fluid energy  (N = {N}, $\\rho$ = {RHO}, T = {T_INIT})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "md_energy.png"), dpi=150)
    plt.close(fig)

    # ---- temperature vs time ------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, T_arr, lw=0.8, color="tab:orange")
    ax.axhline(T_INIT, color="k", ls="--", lw=0.8, label=f"$T_0 = {T_INIT}$")
    ax.set_xlabel(r"time  [$\tau_{LJ}$]")
    ax.set_ylabel(r"$T$  [$\varepsilon / k_B$]")
    ax.set_title("Instantaneous temperature")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "md_temperature.png"), dpi=150)
    plt.close(fig)

    # ---- particle snapshots (t = 0 and t = end) -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, frame, label in zip(axes, [0, -1], ["$t = 0$", f"$t = {times[-1]:.1f}\\,\\tau$"]):
        ax.scatter(traj[frame, :, 0], traj[frame, :, 1], s=12, alpha=0.7)
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel(r"$x\;[\sigma]$")
        ax.set_ylabel(r"$y\;[\sigma]$")
        ax.grid(True, alpha=0.2)
    fig.suptitle(f"LJ fluid snapshots  (N = {N}, $\\rho$ = {RHO})", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "md_snapshots.png"), dpi=150)
    plt.close(fig)

    # ---- radial distribution function (averaged over last 20 % of frames) ---
    n_avg = max(1, traj.shape[0] // 5)
    r_mid, gr_sum = None, None
    for frame in range(traj.shape[0] - n_avg, traj.shape[0]):
        r_mid, gr = _rdf(traj[frame], L)
        gr_sum = gr if gr_sum is None else gr_sum + gr
    gr_avg = gr_sum / n_avg

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r_mid, gr_avg, lw=1.2, color="tab:blue")
    ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.5, label="ideal gas")
    ax.set_xlabel(r"$r\;[\sigma]$")
    ax.set_ylabel(r"$g(r)$")
    ax.set_title(f"Radial distribution function  ($\\rho$ = {RHO}, T ≈ {np.mean(T_arr[-n_avg:]):.2f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "md_rdf.png"), dpi=150)
    plt.close(fig)

    drift = (E_tot - E_tot[0]) / abs(E_tot[0])
    print(f"  max |dE/E0| = {np.max(np.abs(drift)):.2e}")
    print("figures written to", FIGDIR)


if __name__ == "__main__":
    main()
