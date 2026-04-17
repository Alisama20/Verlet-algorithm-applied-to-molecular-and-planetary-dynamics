"""Run the planet-formation toy model and save three figures.

Outputs (all under ``figures/``):

    formation_disk.png     — initial planetesimal disk (coloured by distance)
    formation_final.png    — survivors after merging; symbol area ∝ mass
    formation_mass.png     — cumulative mass distribution of survivors
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from solar_system import simulate_formation

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(FIGDIR, exist_ok=True)


def main():
    print("Running planet-formation simulation ...")
    initial_pos, final_pos, final_mass, n_coll = simulate_formation(
        n_planetesimals=1000, n_steps=1000, dt=1.0, growth_factor=10000.0, seed=2026
    )
    print(f"  {n_coll} merges, {len(final_mass)} survivors")

    # ---- initial disk -------------------------------------------------------
    dist_i = np.hypot(initial_pos[:, 0], initial_pos[:, 1])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    sc = ax.scatter(
        initial_pos[:, 0], initial_pos[:, 1],
        c=dist_i, cmap="plasma", s=2, alpha=0.6,
    )
    plt.colorbar(sc, ax=ax, label="distance [AU]")
    ax.plot(0, 0, "y*", ms=12, label="Sun")
    ax.set_title("Initial planetesimal disk  ($N = 1000$)")
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "formation_disk.png"), dpi=150)
    plt.close(fig)

    # ---- final survivors (size ∝ mass^(1/3) ≈ radius) ----------------------
    dist_f = np.hypot(final_pos[:, 0], final_pos[:, 1])
    m_ref = final_mass.min()
    # Marker area proportional to (mass / m_min)^(2/3) so radius ∝ m^(1/3).
    marker_area = 10.0 * (final_mass / m_ref) ** (2.0 / 3.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    sc = ax.scatter(
        final_pos[:, 0], final_pos[:, 1],
        s=marker_area, c=dist_f, cmap="plasma", alpha=0.7,
    )
    plt.colorbar(sc, ax=ax, label="distance [AU]")
    ax.plot(0, 0, "y*", ms=12, label="Sun")
    ax.set_title(f"After 1000 steps — {len(final_mass)} survivors  ({n_coll} merges)")
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "formation_final.png"), dpi=150)
    plt.close(fig)

    # ---- cumulative mass distribution ---------------------------------------
    m_sorted = np.sort(final_mass)[::-1]
    rank = np.arange(1, len(m_sorted) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(rank, m_sorted / m_ref, "o-", ms=4, lw=1)
    ax.set_xlabel("rank (largest first)")
    ax.set_ylabel(r"mass  [$m_{\min}$]")
    ax.set_title("Cumulative mass distribution of survivors")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "formation_mass.png"), dpi=150)
    plt.close(fig)

    print("figures written to", FIGDIR)


if __name__ == "__main__":
    main()
