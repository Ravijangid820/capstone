"""Per-hospital synthetic scanner shift — the non-IID source.

Each hospital applies a fixed, **nonlinear + spatial** transform emulating its scanner:
gamma (contrast) + a smooth multiplicative bias field + mild blur. Nonlinear/spatial on purpose,
so it **survives per-image z-normalization** (a purely linear intensity shift would be normalized
away, leaving no real heterogeneity). Applied to the 4 modalities only — never the segmentation.
See docs/data-pipeline.md §2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


@dataclass(frozen=True)
class ShiftParams:
    gamma: float       # nonlinear contrast exponent (x^gamma on [0,1])
    bias_amp: float    # +/- amplitude of the smooth multiplicative field
    blur_sigma: float  # gaussian blur sigma (voxels)


# Starting values — H1-H3 are mild, same-direction (they cluster); H4 is the outlier (strong,
# clearly apart). Calibrated so shifts survive z-norm and H4's outlier margin is large; the real
# calibration is whether H2 appears once we train (revisit then). See docs/data-pipeline.md §2.
HOSPITAL_SHIFTS: dict[str, ShiftParams] = {
    "H1": ShiftParams(gamma=1.06, bias_amp=0.06, blur_sigma=0.3),
    "H2": ShiftParams(gamma=1.13, bias_amp=0.09, blur_sigma=0.5),
    "H3": ShiftParams(gamma=1.20, bias_amp=0.12, blur_sigma=0.7),
    "H4": ShiftParams(gamma=1.85, bias_amp=0.34, blur_sigma=1.7),  # <-- outlier
}


def _bias_field(shape: tuple[int, int, int], amp: float, seed: int) -> np.ndarray:
    """A smooth multiplicative field, mean ~1, values in ~[1-amp, 1+amp], fixed per seed."""
    rng = np.random.default_rng(seed)
    coarse = rng.standard_normal((4, 4, 4))                 # low-frequency control points
    factors = tuple(s / c for s, c in zip(shape, coarse.shape))
    field = zoom(coarse, factors, order=3)                  # smooth cubic upsample
    field = field[: shape[0], : shape[1], : shape[2]]
    field = field / (np.abs(field).max() + 1e-8)            # -> [-1, 1]
    return (1.0 + amp * field).astype(np.float32)


def apply_shift(volume: np.ndarray, hospital: str, seed: int = 42) -> np.ndarray:
    """Apply a hospital's scanner shift to a (C, X, Y, Z) multi-modal volume (float).

    The bias field is fixed per (hospital, seed) — a hospital's "scanner" does not change.
    """
    p = HOSPITAL_SHIFTS[hospital]
    hidx = int(hospital[1:])
    out = volume.astype(np.float32).copy()
    field = _bias_field(volume.shape[1:], p.bias_amp, seed * 100 + hidx)

    for c in range(out.shape[0]):
        v = out[c]
        brain = v > 0
        if not brain.any():
            continue
        # 1) gamma: nonlinear contrast over the brain intensity range (background stays 0)
        vmin, vmax = float(v[brain].min()), float(v[brain].max())
        if vmax > vmin:
            norm = np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0)
            norm = np.where(brain, norm ** p.gamma, 0.0)
            v = norm * (vmax - vmin) + vmin
            v = np.where(brain, v, 0.0)
        # 2) bias field: smooth multiplicative gain (spatial)
        v = v * field
        # 3) blur: mild resolution/PSF difference (spatial)
        if p.blur_sigma > 0:
            v = gaussian_filter(v, sigma=p.blur_sigma)
        out[c] = v

    return out
