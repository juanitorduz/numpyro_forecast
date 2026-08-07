"""Draw the three-corruption-patterns schematic (paper Figure 1).

One toy latent demand path is corrupted three ways, one per panel, matching the paper's
three case studies: binary availability gating, a known capacity cap, and fractional
availability attenuation. Styled like the example notebooks (arviz-darkgrid).
"""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).resolve().parents[1] / "figures"

az.style.use("arviz-darkgrid")
plt.rcParams["figure.dpi"] = 200
plt.rcParams["figure.facecolor"] = "white"

DEMAND_COLOR = "black"
SALES_COLOR = "C0"
SHADE_COLOR = "0.75"


def latent_demand(t: np.ndarray) -> np.ndarray:
    """Toy latent demand: level plus weekly cycle plus a slow rise."""
    return 2.0 + 0.012 * t + 0.65 * np.sin(2 * np.pi * t / 7) + 0.25 * np.sin(2 * np.pi * t / 30)


def main() -> None:
    """Render the schematic to ``paper/figures/fig01_corruption_patterns.png``."""
    t = np.arange(60)
    d = latent_demand(t)

    fig, axes = plt.subplots(
        nrows=1, ncols=3, figsize=(12, 3.4), sharey=True, layout="constrained"
    )

    # Panel 1: binary availability gates sales to zero during stockout runs.
    ax = axes[0]
    a_binary = np.ones_like(t, dtype=float)
    for start, stop in [(14, 21), (40, 44)]:
        a_binary[start:stop] = 0.0
        ax.axvspan(start - 0.5, stop - 0.5, color=SHADE_COLOR, alpha=0.5, zorder=0)
    ax.plot(t, d, color=DEMAND_COLOR, lw=1.8, label="latent demand")
    ax.plot(t, a_binary * d, color=SALES_COLOR, lw=1.8, label="observed sales")
    ax.set_title("Binary availability\n(off-shelf days sell zero)", fontsize=12)

    # Panel 2: a known capacity cap clips demand peaks.
    ax = axes[1]
    cap = 2.55
    ax.plot(t, d, color=DEMAND_COLOR, lw=1.8, label="latent demand")
    ax.plot(t, np.minimum(d, cap), color=SALES_COLOR, lw=1.8, label="observed sales")
    ax.axhline(cap, color="C3", ls="--", lw=1.2)
    ax.annotate("capacity cap", xy=(1.5, cap), xytext=(1.5, cap + 0.25), color="C3", fontsize=9)
    ax.set_title("Known capacity cap\n(peaks record the cap)", fontsize=12)

    # Panel 3: fractional availability attenuates sales toward a small floor.
    ax = axes[2]
    a_frac = np.ones_like(t, dtype=float)
    a_frac[10:18] = np.array([0.85, 0.6, 0.35, 0.2, 0.3, 0.55, 0.75, 0.9])
    a_frac[33:42] = np.array([0.7, 0.45, 0.25, 0.1, 0.05, 0.2, 0.4, 0.65, 0.85])
    floor = 0.05
    f = floor + (1 - floor) * a_frac
    low = a_frac < 1.0
    ax.fill_between(t, 0, 3.6, where=low, color=SHADE_COLOR, alpha=0.5, step="mid", zorder=0)
    ax.plot(t, d, color=DEMAND_COLOR, lw=1.8, label="latent demand")
    ax.plot(t, f * d, color=SALES_COLOR, lw=1.8, label="observed sales")
    ax.set_title("Fractional availability\n(partial-day stockouts attenuate)", fontsize=12)

    for ax in axes:
        ax.set_xlabel("time")
        ax.set_ylim(-0.15, 3.6)
    axes[0].set_ylabel("units")
    axes[0].legend(loc="lower right", fontsize=9, framealpha=0.9)

    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "fig01_corruption_patterns.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
