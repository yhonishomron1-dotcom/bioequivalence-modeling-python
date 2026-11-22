"""Section 5 - Two-compartment lung regional deposition model.

dU/dt = -kUA*U ;  dA/dt = kUA*U - kAE*A   (upper airways U, alveoli A)
Solved numerically and checked against the closed-form solution.
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

kUA, kAE = 0.3, 0.1          # 1/min: upper->alveoli transfer, alveolar elimination
U0, A0 = 1.0, 0.0            # dose initially deposited in upper airways


def rhs(t, y):
    U, A = y
    return [-kUA * U, kUA * U - kAE * A]


def analytic_A(t):
    return kUA * U0 / (kAE - kUA) * (np.exp(-kUA * t) - np.exp(-kAE * t))


def main():
    t = np.linspace(0, 60, 400)
    sol = solve_ivp(rhs, (0, 60), [U0, A0], t_eval=t, rtol=1e-9, atol=1e-12)
    U, A = sol.y
    err = np.max(np.abs(A - analytic_A(t)))
    t_pk = np.log(kUA / kAE) / (kUA - kAE)
    print(f"Peak alveolar fraction : {A.max():.3f} at t = {t[A.argmax()]:.1f} min "
          f"(analytic t_peak = {t_pk:.2f} min)")
    print(f"Fraction at 60 min     : upper airways {U[-1]:.4f}, alveoli {A[-1]:.4f}")
    print(f"Max |numeric - analytic| = {err:.2e}")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(t, U, "b-", lw=2, label="Upper airways")
    ax.plot(t, A, "r--", lw=2, label="Alveoli")
    ax.set(xlabel="Time (min)", ylabel="Fraction of dose", title="Simulated lung regional deposition")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_05_lung.png", dpi=150)


if __name__ == "__main__":
    main()
