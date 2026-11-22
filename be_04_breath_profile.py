"""Section 4 - Patient breath profiling and simulation.

Simulates an inhalation waveform, then runs the same analysis on a noisy
'measured' copy (smoothing + peak/timing extraction), as you would with spirometry data.
Requires: numpy scipy matplotlib
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)


def analyse(t, flow, fs, thresh=1e-3):
    """Peak flow, inhalation time and inhaled volume (arbitrary units)."""
    inhal = flow > thresh
    return flow.max(), inhal.sum() / fs, np.trapezoid(flow, t)


def main():
    fs = 100                                     # Hz
    t = np.arange(0, 5 + 1 / fs, 1 / fs)         # 5 s
    rate = 12 / 60                               # 12 breaths/min = 0.2 Hz
    flow = np.maximum(0, np.sin(2 * np.pi * rate * t))

    peak, tin, vol = analyse(t, flow, fs)
    print(f"Simulated  -> peak flow {peak:.3f}, inhalation time {tin:.2f} s, volume {vol:.3f}")

    rng = np.random.default_rng(1)
    noisy = flow + rng.normal(0, 0.05, flow.size)
    smooth = np.maximum(0, savgol_filter(noisy, 31, 3))
    peak, tin, vol = analyse(t, smooth, fs, thresh=0.05)
    print(f"'Measured' -> peak flow {peak:.3f}, inhalation time {tin:.2f} s, "
          f"volume {vol:.3f}")

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(t, noisy, color="#B4B2A9", lw=1, label="Noisy 'measured' signal")
    ax.plot(t, smooth, color="#378ADD", lw=1.5, label="Smoothed")
    ax.plot(t, flow, "k--", lw=1.5, label="Simulated ideal")
    ax.set(xlabel="Time (s)", ylabel="Flow rate (arbitrary units)", title="Simulated patient inhalation profile")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_04_breath.png", dpi=150)


if __name__ == "__main__":
    main()
