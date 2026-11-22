"""Section 3 - In-vitro release testing (IVRT): reference vs generic.

AUC, the simple two-sample t-test from the original outline, and the f2
similarity factor (the usual regulatory metric for comparing release profiles).
Requires: numpy scipy matplotlib
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

time = np.array([0, 5, 10, 15, 20, 30], float)
ref = np.array([0, 10, 22, 35, 48, 60], float)
gen = np.array([0, 9, 20, 34, 46, 59], float)


def f2(r, t):
    """f2 similarity factor; f2 >= 50 is conventionally 'similar'.
    Uses points after t = 0 (here all 5 are below the 85% cap)."""
    d = r[1:] - t[1:]
    return 50 * np.log10(100 / np.sqrt(1 + np.mean(d ** 2)))


def main():
    auc_r, auc_g = np.trapezoid(ref, time), np.trapezoid(gen, time)
    t, p = stats.ttest_ind(ref, gen)                    # same as MATLAB ttest2
    print(f"AUC Reference : {auc_r:.2f}")
    print(f"AUC Generic   : {auc_g:.2f}   (ratio {auc_g / auc_r:.3f})")
    print(f"t-test p-value: {p:.4f}")
    f2v = f2(ref, gen)
    print(f"f2 similarity : {f2v:.1f}  ({'similar' if f2v >= 50 else 'not similar'}, threshold 50)")
    print("Note: the t-test treats time points as independent replicates; with real")
    print("      data use f2 or compare replicate vessels (n >= 12) instead.")

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(time, ref, "b-o", label="Reference")
    ax.plot(time, gen, "r-s", label="Generic")
    ax.set(xlabel="Time (min)", ylabel="Cumulative % release", title="In-vitro release testing")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_03_ivrt.png", dpi=150)


if __name__ == "__main__":
    main()
