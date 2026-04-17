"""Integrate the 8-planet Solar System, save three figures and print the
orbital-period errors.

Outputs (all under ``figures/``):

    solar_inner.png   — inner planets (Mercury, Venus, Earth, Mars)
    solar_outer.png   — outer planets (Jupiter .. Neptune)
    solar_energy.png  — relative drift of the total mechanical energy
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from solar_system import (
    simulate,
    solar_system_ic,
    PLANET_NAMES,
    REAL_PERIODS_DAYS,
)
from solar_system.verlet import measure_periods


FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(FIGDIR, exist_ok=True)


def main():
    mass, pos, vel = solar_system_ic()

    print("Integrating N = %d bodies ..." % len(mass))
    traj, energy, times = simulate(mass, pos, vel, dt=1e-3, n_steps=200_000,
                                   save_every=50)
    print("  %d frames recorded" % traj.shape[0])

    # --- orbital periods --------------------------------------------------
    periods = measure_periods(traj, times)
    print()
    print("%-10s %10s %10s %8s" % ("body", "period_num", "period_real", "err %"))
    for i in range(1, len(mass)):
        p = periods[i]
        real = REAL_PERIODS_DAYS[i]
        if np.isnan(p):
            print("%-10s %10s %10.1f %8s" % (PLANET_NAMES[i], "n/a", real, "n/a"))
        else:
            err = abs(p - real) / real * 100.0
            print("%-10s %10.1f %10.1f %8.2f" % (PLANET_NAMES[i], p, real, err))

    # --- trajectories (inner planets) ------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_title("Inner solar system")
    for i in range(0, 5):
        ax.plot(traj[:, i, 0], traj[:, i, 1], "-", lw=0.8,
                label=PLANET_NAMES[i])
    ax.plot(0, 0, "y*", ms=14)
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "solar_inner.png"), dpi=150)
    plt.close(fig)

    # --- trajectories (outer planets) -------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_title("Outer solar system")
    for i in range(5, 9):
        ax.plot(traj[:, i, 0], traj[:, i, 1], "-", lw=0.8,
                label=PLANET_NAMES[i])
    ax.plot(0, 0, "y*", ms=14)
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "solar_outer.png"), dpi=150)
    plt.close(fig)

    # --- energy conservation ---------------------------------------------
    drift = (energy - energy[0]) / abs(energy[0])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, drift)
    ax.set_xlabel("time [natural units  ~58.1 d]")
    ax.set_ylabel(r"$(E(t) - E_0) / |E_0|$")
    ax.set_title("Relative drift of the total energy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "solar_energy.png"), dpi=150)
    plt.close(fig)

    print()
    print("max |dE/E0| =", float(np.max(np.abs(drift))))
    print("figures written to", FIGDIR)


if __name__ == "__main__":
    main()
