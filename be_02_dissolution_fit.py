"""Section 2 - Nasal-spray API particle size and dissolution kinetics.

Fits Weibull and Higuchi models to dissolution data and simulates how particle
size could shift the profile (illustrative assumption: Weibull scale ~ diameter).
Requires: numpy scipy matplotlib
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

time = np.array([0, 5, 10, 15, 20, 30, 45, 60], float)        # min
dissolution = np.array([0, 12, 25, 40, 55, 70, 85, 95], float)  # %


def weibull(t, a, b):
    return 100 * (1 - np.exp(-(t / a) ** b))


def higuchi(t, k):
    return k * np.sqrt(t)


def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    (a, b), _ = curve_fit(weibull, time, dissolution, p0=[20, 1])
    (k,), _ = curve_fit(higuchi, time, dissolution, p0=[10])
    rmse_w = np.sqrt(np.mean((weibull(time, a, b) - dissolution) ** 2))
    rmse_h = np.sqrt(np.mean((higuchi(time, k) - dissolution) ** 2))
    r2_w, r2_h = r2(dissolution, weibull(time, a, b)), r2(dissolution, higuchi(time, k))
    print(f"Weibull fit : a = {a:.2f} min, b = {b:.3f}, R2 = {r2_w:.4f}, RMSE = {rmse_w:.2f}")
    print(f"Higuchi fit : k = {k:.2f} %/min^0.5, R2 = {r2_h:.4f}, RMSE = {rmse_h:.2f}")

    tf = np.linspace(0, 60, 200)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(time, dissolution, "ro", label="Experimental")
    ax[0].plot(tf, weibull(tf, a, b), "b-", lw=2, label="Weibull fit")
    ax[0].plot(tf, higuchi(tf, k), "g--", lw=1.5, label="Higuchi fit")
    ax[0].set(xlabel="Time (min)", ylabel="Dissolution (%)", title="Dissolution profile fit")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    for scale, ls in [(0.5, ":"), (1.0, "-"), (2.0, "--")]:      # relative particle diameter
        ax[1].plot(tf, weibull(tf, a * scale, b), ls, lw=2, label=f"{scale:g}x diameter")
        t50 = a * scale * np.log(2) ** (1 / b)
        print(f"Particle size {scale:g}x: time to 50% dissolved = {t50:.1f} min")
    ax[1].set(xlabel="Time (min)", ylabel="Dissolution (%)", title="Illustrative particle-size effect")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_02_dissolution.png", dpi=150)


if __name__ == "__main__":
    main()
