"""Section 6 - Basic PBPK model (lung -> plasma), Python equivalent of the SimBiology example.

Compartments: Lung (1 L) and Plasma (3 L); first-order absorption k_abs and elimination k_elim.
  dLung/dt   = -k_abs*Lung
  dPlasma/dt =  k_abs*Lung - k_elim*Plasma
Amounts are in dose units; concentration = amount / volume.
Requires: numpy scipy matplotlib
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

V_LUNG, V_PLASMA = 1.0, 3.0       # L
K_ABS, K_ELIM = 0.2, 0.1          # 1/min
DOSE = 1.0


def simulate(k_abs=K_ABS, k_elim=K_ELIM, stop=60, n=600):
    rhs = lambda t, y: [-k_abs * y[0], k_abs * y[0] - k_elim * y[1]]
    t = np.linspace(0, stop, n)
    sol = solve_ivp(rhs, (0, stop), [DOSE, 0.0], t_eval=t, rtol=1e-9, atol=1e-12)
    return t, sol.y[0], sol.y[1]


def main():
    t, lung, plasma = simulate()
    conc = plasma / V_PLASMA
    i = conc.argmax()
    print(f"Cmax = {conc[i]:.4f} (dose units/L) at Tmax = {t[i]:.1f} min")
    print(f"AUC(0-60 min) of plasma concentration = {np.trapezoid(conc, t):.3f}")
    print(f"Plasma fraction of dose remaining at 60 min = {plasma[-1]:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(t, lung / V_LUNG, "b-", lw=2, label="Lung")
    ax[0].plot(t, conc, "r--", lw=2, label="Plasma")
    ax[0].set(xlabel="Time (min)", ylabel="Concentration (dose/L)", title="Simulated PBPK profiles")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    for ka in (0.1, 0.2, 0.4):                       # sensitivity to absorption rate
        tt, _, pl = simulate(k_abs=ka)
        ax[1].plot(tt, pl / V_PLASMA, lw=2, label=f"k_abs = {ka}")
    ax[1].set(xlabel="Time (min)", ylabel="Plasma conc. (dose/L)", title="Sensitivity to absorption rate")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_06_pbpk.png", dpi=150)


if __name__ == "__main__":
    main()
