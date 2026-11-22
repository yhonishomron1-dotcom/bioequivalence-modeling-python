"""Section 1 - Morphologically-directed Raman spectroscopy (MDRS) image analysis.

Segments particles in a micrograph and measures size and morphology.
Usage:  python be_01_mdrs_particle_analysis.py [particle_image.tif]
With no argument a synthetic micrograph is generated so the script runs as-is.
Requires: numpy scipy scikit-image matplotlib
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import color, filters, io, measure

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)


def synthetic_image(seed=0, size=512, n=90):
    """Dark, slightly elongated particles on an unevenly lit background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    img = 0.80 + 0.10 * (xx / size)                      # illumination gradient
    for _ in range(n):
        cx, cy = rng.uniform(20, size - 20, 2)
        r = rng.lognormal(np.log(8), 0.35)
        e = rng.uniform(1.0, 1.6)
        th = rng.uniform(0, np.pi)
        x, y = xx - cx, yy - cy
        u, v = x * np.cos(th) + y * np.sin(th), -x * np.sin(th) + y * np.cos(th)
        img -= 0.55 * (((u / (r * e)) ** 2 + (v / r) ** 2) <= 1)
    img = ndi.gaussian_filter(img, 1.0) + rng.normal(0, 0.02, img.shape)
    return np.clip(img, 0, 1)


def main():
    img = io.imread(sys.argv[1]) if len(sys.argv) > 1 else synthetic_image()
    gray = color.rgb2gray(img) if img.ndim == 3 else img
    gray = gray / gray.max()

    # Adaptive threshold: particles are darker than their local surroundings
    bw = gray < filters.threshold_local(gray, block_size=51, offset=0.05)
    lab0 = measure.label(bw)                             # remove noise (< 50 px), like bwareaopen
    keep = np.bincount(lab0.ravel()) >= 50
    keep[0] = False
    bw = ndi.binary_fill_holes(keep[lab0])               # fill holes
    labels = measure.label(bw)

    props = measure.regionprops(labels)
    area = np.array([p.area for p in props])
    perim = np.array([p.perimeter_crofton for p in props])
    ecc = np.array([p.eccentricity for p in props])
    circ = np.clip(4 * np.pi * area / perim ** 2, 0, 1)

    print(f"Particles detected      : {len(props)}")
    print(f"Median area (pixels)    : {np.median(area):.1f}")
    print(f"Mean particle circularity: {circ.mean():.3f}")
    print(f"Mean eccentricity       : {ecc.mean():.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    ax[0].imshow(gray, cmap="gray"); ax[0].set_title("Micrograph")
    ax[1].imshow(color.label2rgb(labels, gray, alpha=0.35, bg_label=0))
    ax[1].set_title(f"Segmented ({len(props)} particles)")
    ax[2].hist(area, bins=20, color="#1D9E75")
    ax[2].set(xlabel="Area (pixels)", ylabel="Count", title="Particle size distribution")
    for a in ax[:2]:
        a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "fig_01_mdrs.png", dpi=150)


if __name__ == "__main__":
    main()
