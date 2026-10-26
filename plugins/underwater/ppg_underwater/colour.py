"""Colours under water: a physics-based restoration with the range of every pixel, and a quick
correction for the photos before they are aligned.

**Restoration** (after Sea-thru: D. Akkaynak and T. Treibitz, *Sea-thru: A Method for Removing Water
From Underwater Images*, CVPR 2019; written here from the paper). A pixel of colour channel ``c``
at range ``z`` (metres along its ray) is

    I_c = J_c * exp(-beta_D_c(z) * z) + B_inf_c * (1 - exp(-beta_B_c * z))

- the light of the surface ``J`` attenuated along the way (``beta_D``, larger for red, depending on
the range itself) plus the light the water scatters back towards the camera, growing with the range
up to ``B_inf`` (the colour of the water far away). The steps:

1. *Backscatter.* The range is cut into bins; in every bin the darkest pixels (the lowest percentile
   of R + G + B) are almost only backscatter, a little of the darkest surfaces. ``B_c(z) = B_inf_c (1 -
   exp(-beta_B_c z)) + J'_c exp(-beta_D'_c z)`` is fitted to them, per channel, by bounded least
   squares; the backscatter of every pixel is subtracted: ``D = I - B(z)``.
2. *Illuminant.* What is left is the light reaching the camera from the surface; its local average
   among pixels at a similar range (the local space average colour of the paper, computed here by a
   range-aware Gaussian average) is the illuminant ``E_c(x)`` reaching that range.
3. *Attenuation.* ``log E_c = w_c + e_p - beta_D_c(z) z``, with ``beta_D_c(z) = a exp(b z) + c exp(d z)``
   (the paper's two-term model; a constant when the photos do not span ranges enough), ``w_c`` the
   white of the light and ``e_p`` the exposure of each photo, fitted by alternating least squares.
4. *Restored:* ``J_c = D_c exp(beta_D_c(z) z) / (exp(w_c) e_p)``, scaled so that the median grey of
   the survey is a middle grey, and written as sRGB.

The coefficients are fitted on one photo at a time, or on a sample of the photos of the survey
together (``per="survey"``, the default: the water is the same, so the photos of one dive get the
same correction and the orthomosaic and the texture have no seams of colour; the exposure of every
photo and the strength of its backscatter stay its own). The model works on linear light: the 8-bit
sRGB of the photos is decoded first and encoded again at the end.

**Quick correction** (no range: before the alignment, or without depth maps), after C. O. Ancuti et
al., *Color Balance and Fusion for Underwater Image and Video Enhancement*, IEEE TIP 2018: the red
channel (and, in green water, the blue one) compensated with the green one,
``I_rc = I_r + alpha (mean_g - mean_r) (1 - I_r) I_g``, a grey-world white balance, a stretch of the
lightness and optionally CLAHE (local contrast). The means and gains come from a sample of the
photos together, so that the photos of a survey stay alike.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

WORK_SIDE = 900  # long side of the copies the coefficients are fitted on
MAX_GAIN = 12.0  # most the attenuation is undone (far pixels: noise)


# ------------------------------------------------------------------ light
def to_linear(img: np.ndarray) -> np.ndarray:
    """sRGB 8 or 16 bit (or float 0-1) to linear light, float32 0-1."""
    a = img.astype(np.float32)
    if img.dtype == np.uint8:
        a /= 255.0
    elif img.dtype == np.uint16:
        a /= 65535.0
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4).astype(np.float32)


def to_srgb(lin: np.ndarray) -> np.ndarray:
    """Linear light 0-1 to 8-bit sRGB."""
    a = np.clip(lin, 0.0, 1.0)
    s = np.where(a <= 0.0031308, 12.92 * a, 1.055 * np.power(a, 1 / 2.4) - 0.055)
    return np.clip(np.rint(s * 255.0), 0, 255).astype(np.uint8)


def reduced(img: np.ndarray, side: int = WORK_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    f = min(1.0, side / max(h, w))
    if f >= 1.0:
        return img
    return cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)


def as_colour(img: np.ndarray) -> np.ndarray:
    """A BGR image (a monochrome one repeated in the three channels)."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return img[..., :3]
    return img


# ------------------------------------------------------------------ backscatter
def dark_samples(lin: np.ndarray, rng: np.ndarray, valid: np.ndarray, bins: int = 10, fraction: float = 0.01,
                 lo: float | None = None, hi: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(z, rgb)`` of the darkest pixels (the lowest ``fraction`` of R + G + B) of every range bin."""
    z = rng[valid]
    v = lin[valid]
    if not len(z):
        return np.zeros(0), np.zeros((0, 3))
    lo = float(np.percentile(z, 1)) if lo is None else lo
    hi = float(np.percentile(z, 99)) if hi is None else hi
    edges = np.linspace(lo, hi, bins + 1)
    zs, vs = [], []
    s = v.sum(axis=1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (z >= a) & (z < b)
        n = int(m.sum())
        if n < 50:
            continue
        k = max(5, int(n * fraction))
        idx = np.flatnonzero(m)
        pick = idx[np.argpartition(s[idx], k - 1)[:k]]
        zs.append(z[pick])
        vs.append(v[pick])
    if not zs:
        return np.zeros(0), np.zeros((0, 3))
    return np.concatenate(zs), np.concatenate(vs)


def backscatter_model(z, p) -> np.ndarray:
    binf, bb, jp, bd = p
    return binf * (1.0 - np.exp(-bb * z)) + jp * np.exp(-bd * z)


def fit_backscatter(z: np.ndarray, values: np.ndarray, zref: float) -> list[list[float]]:
    """``[B_inf, beta_B, J', beta_D']`` per channel (betas per unit of ``z / zref``)."""
    from scipy.optimize import least_squares

    zn = np.asarray(z, float) / zref
    out = []
    for c in range(3):
        y = np.asarray(values[:, c], float)
        best = None
        for bb0 in (0.3, 1.0, 3.0):
            for bd0 in (0.5, 3.0):
                x0 = [min(max(float(np.max(y)) * 1.2, 1e-3), 1.0), bb0, min(float(np.min(y)), 1.0) * 0.5, bd0]
                try:
                    res = least_squares(lambda p: backscatter_model(zn, p) - y, x0,
                                        bounds=([0, 0, 0, 0], [1, 50, 1, 50]), loss="soft_l1", f_scale=0.02,
                                        max_nfev=400)
                except ValueError:
                    continue
                if best is None or res.cost < best.cost:
                    best = res
        out.append([float(v) for v in best.x] if best is not None else [float(np.median(y)), 1.0, 0.0, 1.0])
    return out


def backscatter(rng: np.ndarray, params, zref: float, amplitude: float = 1.0) -> np.ndarray:
    zn = rng / zref
    return np.stack([amplitude * backscatter_model(zn, params[c]) for c in range(3)], axis=-1).astype(np.float32)


# ------------------------------------------------------------------ illuminant and attenuation
def local_illuminant(direct: np.ndarray, rng: np.ndarray, valid: np.ndarray, layers: int = 16,
                     sigma: float = 0.08) -> np.ndarray:
    """The local average of ``direct`` among the pixels at a similar range: Gaussian in the image
    (``sigma`` of the long side) and in the range (``layers`` between the 2nd and 98th percentiles,
    each holding the pixels within about a third of the spacing: averaging over a wider span of
    ranges flattens the attenuation, the far pixels lifted by the near ones)."""
    h, w = rng.shape
    s = max(2.0, sigma * max(h, w))
    z = rng[valid]
    lo, hi = np.percentile(z, [2, 98]) if len(z) else (0.0, 1.0)
    hi = max(hi, lo + 1e-6)
    centres = np.linspace(lo, hi, layers)
    sz = (hi - lo) / max(layers - 1, 1) / 1.5
    num = np.zeros_like(direct)
    den = np.zeros((h, w), np.float32)
    vf = valid.astype(np.float32)
    for zc in centres:
        wk = np.exp(-0.5 * ((rng - zc) / sz) ** 2).astype(np.float32) * vf
        bw = cv2.GaussianBlur(wk, (0, 0), s)
        bd = cv2.GaussianBlur(direct * wk[..., None], (0, 0), s)
        avg = bd / np.maximum(bw, 1e-6)[..., None]
        num += avg * wk[..., None]
        den += wk
    return num / np.maximum(den, 1e-6)[..., None]


def beta_model(zn, p) -> np.ndarray:
    a, b, c, d = p
    return a * np.exp(b * zn) + c * np.exp(d * zn)


def fit_attenuation(zn: np.ndarray, log_e: np.ndarray, photo: np.ndarray, photos: int, two_terms: bool = True,
                    rounds: int = 8) -> dict:
    """Alternating fit of ``log E = w_c + e_p - beta_c(z) z`` (see the module) on samples at
    normalised ranges ``zn``. Returns ``{"beta": [[a, b, c, d]] * 3, "white": [w_c], "exposure":
    [e_p], "two_terms"}``."""
    from scipy.optimize import least_squares

    zn = np.asarray(zn, float)
    y = np.asarray(log_e, float)
    photo = np.asarray(photo, int)
    eps = np.zeros(photos)
    white = np.zeros(3)
    # a constant beta from the slope of log E over the range, the photo intercepts apart
    beta = []
    for c in range(3):
        a_rows = np.column_stack([np.eye(photos)[photo], -zn])
        sol, *_ = np.linalg.lstsq(a_rows, y[:, c], rcond=None)
        beta.append([max(float(sol[-1]), 1e-3), 0.0, 0.0, 0.0])
    span = float(np.percentile(zn, 95) / max(np.percentile(zn, 5), 1e-6))
    use_two = two_terms and span > 1.6
    for _ in range(rounds):
        # intercepts per photo and channel, split into a white per channel and an exposure per photo
        att = np.column_stack([beta_model(zn, beta[c]) * zn for c in range(3)])
        r = y + att
        table = np.zeros((photos, 3))
        for p in range(photos):
            m = photo == p
            if m.any():
                table[p] = np.median(r[m], axis=0)
        present = np.array([(photo == p).any() for p in range(photos)])
        for _k in range(5):
            white = np.median(table[present] - eps[present][:, None], axis=0)
            eps[present] = np.median(table[present] - white[None, :], axis=1)
            eps -= np.median(eps[present]) if present.any() else 0.0
        # the attenuation of every channel on its target
        for c in range(3):
            target = white[c] + eps[photo] - y[:, c]

            def resid(p, c=c, target=target):
                return beta_model(zn, p) * zn - target

            x0 = beta[c] if use_two else [beta[c][0], 0.0, 0.0, 0.0]
            if use_two and x0[2] == 0.0:
                x0 = [x0[0], -0.1, 0.1 * x0[0] + 1e-3, -1.0]
            lo = [0.0, -10.0, 0.0, -10.0] if use_two else [0.0, -1e-9, 0.0, -1e-9]
            hi = [50.0, 0.0, 50.0, 0.0] if use_two else [50.0, 0.0, 1e-9, 0.0]
            x0 = np.clip(x0, np.array(lo) + 1e-12, np.array(hi) - 1e-12)
            try:
                res = least_squares(resid, x0, bounds=(lo, hi), loss="soft_l1", f_scale=0.1, max_nfev=300)
                beta[c] = [float(v) for v in res.x]
            except ValueError:
                pass
    return {"beta": beta, "white": [float(v) for v in white], "exposure": [float(v) for v in eps],
            "two_terms": bool(use_two)}


def attenuation_gain(rng: np.ndarray, beta, zref: float, max_gain: float = MAX_GAIN) -> np.ndarray:
    zn = rng / zref
    g = np.stack([np.exp(beta_model(zn, beta[c]) * zn) for c in range(3)], axis=-1)
    return np.minimum(g, max_gain).astype(np.float32)


# ------------------------------------------------------------------ the model of a survey
@dataclass
class Model:
    """The water of a survey (or of one photo): see the module. Ranges normalised by ``zref``."""

    zref: float
    backscatter: list  # [[B_inf, beta_B, J', beta_D']] per channel (B, G, R order of OpenCV)
    beta: list  # [[a, b, c, d]] per channel
    white: list  # log white per channel
    gain: float = 1.0  # output gain (linear)
    zmin: float = 0.0
    zmax: float = 1.0
    metric: bool = False
    per: str = "survey"
    photos: int = 0
    amplitudes: dict = field(default_factory=dict)  # backscatter amplitude per camera key
    exposures: dict = field(default_factory=dict)  # log exposure per camera key
    two_terms: bool = False
    max_gain: float = MAX_GAIN

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        known["amplitudes"] = {int(k): float(v) for k, v in (d.get("amplitudes") or {}).items()}
        known["exposures"] = {int(k): float(v) for k, v in (d.get("exposures") or {}).items()}
        return cls(**known)

    def coefficients(self) -> dict:
        """Coefficients per metre (when the ranges are metres) for a summary: ``beta_B`` and the
        ``beta_D`` at the median range, per channel R, G, B."""
        unit = self.zref
        out = {}
        for name, c in (("R", 2), ("G", 1), ("B", 0)):
            bd = float(beta_model(1.0, self.beta[c])) / unit
            out[name] = {"B_inf": self.backscatter[c][0], "beta_B": self.backscatter[c][1] / unit, "beta_D": bd}
        return out


def _amplitude(lin, rng, valid, model: Model) -> float | None:
    """The strength of the backscatter of one photo against the survey's: the median ratio of its
    darkest pixels to the model."""
    z, v = dark_samples(lin, np.clip(rng, model.zmin, model.zmax), valid)
    if len(z) < 20:
        return None
    pred = backscatter(z[:, None], model.backscatter, model.zref)[:, 0, :]
    ratio = v.sum(axis=1) / np.maximum(pred.sum(axis=1), 1e-6)
    return float(np.clip(np.median(ratio), 0.2, 5.0))


def restore_linear(lin: np.ndarray, rng: np.ndarray, model: Model, amplitude: float = 1.0,
                   exposure: float = 0.0, rows: int = 512) -> np.ndarray:
    """Restored linear colours of a photo (``lin``, BGR linear) with its ranges ``rng`` (same size)."""
    out = np.empty_like(lin, dtype=np.float32)
    scale = model.gain / np.exp(np.asarray(model.white, np.float32) + exposure)
    for r0 in range(0, lin.shape[0], rows):
        z = np.clip(rng[r0:r0 + rows], model.zmin, model.zmax)
        d = lin[r0:r0 + rows] - backscatter(z, model.backscatter, model.zref, amplitude)
        d = np.maximum(d, 0.0)
        out[r0:r0 + rows] = d * attenuation_gain(z, model.beta, model.zref, model.max_gain) * scale
    return out


def fit_model(samples, per: str = "survey", metric: bool = False, two_terms: bool = True) -> Model:
    """The model of the water from ``samples``: ``[(key, lin, rng, valid)]`` (reduced copies of the
    photos in linear light, BGR, with their ranges and where the ranges are measured)."""
    zs = np.concatenate([r[v] for _k, _l, r, v in samples]) if samples else np.zeros(0)
    if len(zs) < 100:
        raise ValueError("too few pixels with a range")
    zref = float(np.median(zs))
    zmin, zmax = (float(v) for v in np.percentile(zs, [0.5, 99.5]))
    # 1. backscatter: pooled, then the strength of every photo, then pooled again on the normalised samples
    dz, dv, dk = [], [], []
    for i, (_key, lin, rng, valid) in enumerate(samples):
        z, v = dark_samples(lin, rng, valid, lo=zmin, hi=zmax)
        dz.append(z)
        dv.append(v)
        dk.append(np.full(len(z), i))
    z_all, v_all, k_all = np.concatenate(dz), np.concatenate(dv), np.concatenate(dk)
    params = fit_backscatter(z_all, v_all, zref)
    amps = np.ones(len(samples))
    if len(samples) > 1:
        for _round in range(2):
            pred = np.column_stack([backscatter_model(z_all / zref, params[c]) for c in range(3)])
            for i in range(len(samples)):
                m = k_all == i
                if m.sum() >= 20:
                    amps[i] = float(np.clip(np.median(v_all[m].sum(1) / np.maximum(pred[m].sum(1), 1e-6)), 0.2, 5.0))
            amps /= np.median(amps)
            params = fit_backscatter(z_all, v_all / amps[k_all][:, None], zref)
    # 2. illuminant samples of every photo
    rs = np.random.default_rng(0)
    zz, ee, pp = [], [], []
    for i, (_key, lin, rng, valid) in enumerate(samples):
        z = np.clip(rng, zmin, zmax)
        direct = np.maximum(lin - backscatter(z, params, zref, amps[i]), 1e-4)
        light = local_illuminant(direct, z, valid)
        ok = valid & np.all(light > 1e-3, axis=-1) & np.all(lin < 0.98, axis=-1)
        idx = np.flatnonzero(ok)
        if len(idx) > 4000:
            idx = rs.choice(idx, 4000, replace=False)
        zz.append(z.ravel()[idx])
        ee.append(np.log(light.reshape(-1, 3)[idx]))
        pp.append(np.full(len(idx), i))
    zn = np.concatenate(zz) / zref
    att = fit_attenuation(zn, np.concatenate(ee), np.concatenate(pp), len(samples), two_terms=two_terms)
    model = Model(zref=zref, backscatter=params, beta=att["beta"], white=att["white"], zmin=zmin, zmax=zmax,
                  metric=metric, per=per, photos=len(samples), two_terms=att["two_terms"])
    model.amplitudes = {int(k): float(a) for (k, *_r), a in zip(samples, amps)}
    model.exposures = {int(k): float(e) for (k, *_r), e in zip(samples, att["exposure"])}
    # 3. the output gain: the median grey of the survey to a middle grey (linear 0.18)
    greys = []
    for (key, lin, rng, valid) in samples:
        j = restore_linear(lin, rng, model, model.amplitudes[int(key)], model.exposures[int(key)])
        greys.append(np.median(j[valid].mean(axis=1)) if valid.any() else np.nan)
    g = float(np.nanmedian(greys)) if np.isfinite(np.nanmedian(greys)) else 1.0
    model.gain = 0.18 / max(g, 1e-6)
    return model


# ------------------------------------------------------------------ quick correction
@dataclass
class QuickParams:
    alpha: float = 1.0  # strength of the red compensation (0: none)
    blue: float = 0.0  # strength of the blue compensation (green water)
    balance: bool = True  # grey-world white balance
    stretch: bool = True  # lightness stretched between the 0.5th and 99.5th percentiles of the survey
    clahe: bool = False
    clip: float = 2.0  # CLAHE clip limit

    def to_dict(self) -> dict:
        return asdict(self)


def quick_stats(images, params: QuickParams) -> dict:
    """The means, gains and stretch of the quick correction from a sample of photos (BGR uint8,
    reduced copies) together."""
    pix = np.concatenate([as_colour(im).reshape(-1, 3) for im in images]).astype(np.float32) / 255.0
    means = pix.mean(axis=0)  # B, G, R
    comp = _compensate(pix[None], params, means)[0]
    cm = comp.mean(axis=0)
    gains = (cm.mean() / np.maximum(cm, 1e-4)) if params.balance else np.ones(3)
    balanced = np.clip(comp * gains, 0, 1)
    lum = balanced @ np.array([0.114, 0.587, 0.299], np.float32)
    lo, hi = (float(v) for v in np.percentile(lum, [0.5, 99.5])) if params.stretch else (0.0, 1.0)
    return {"means": [float(v) for v in means], "gains": [float(v) for v in gains], "low": lo,
            "high": max(hi, lo + 0.05)}


def _compensate(a: np.ndarray, params: QuickParams, means) -> np.ndarray:
    b, g, r = a[..., 0], a[..., 1], a[..., 2]
    mb, mg, mr = means
    out = a.copy()
    if params.alpha:
        out[..., 2] = r + params.alpha * max(mg - mr, 0.0) * (1.0 - r) * g
    if params.blue:
        out[..., 0] = b + params.blue * max(mg - mb, 0.0) * (1.0 - b) * g
    return out


def quick_correct(img: np.ndarray, params: QuickParams, stats: dict) -> np.ndarray:
    """The quick correction of a photo (BGR uint8; a monochrome one only stretched)."""
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        a = (img.reshape(img.shape[:2]).astype(np.float32) / 255.0 - stats["low"]) / (stats["high"] - stats["low"])
        out = (np.clip(a, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        if params.clahe:
            out = cv2.createCLAHE(clipLimit=params.clip, tileGridSize=(8, 8)).apply(out)
        return out
    a = as_colour(img).astype(np.float32) / 255.0
    a = _compensate(a, params, stats["means"])
    a = np.clip(a * np.asarray(stats["gains"], np.float32), 0, 1)
    if params.stretch:
        lum = a @ np.array([0.114, 0.587, 0.299], np.float32)
        new = np.clip((lum - stats["low"]) / (stats["high"] - stats["low"]), 0, 1)
        a = np.clip(a * (new / np.maximum(lum, 1e-4))[..., None], 0, 1)
    out = (a * 255.0 + 0.5).astype(np.uint8)
    if params.clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=params.clip, tileGridSize=(8, 8)).apply(lab[..., 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return out


# ------------------------------------------------------------------ measures
def neutral_error(img_bgr: np.ndarray) -> float:
    """How far from grey a patch is: the mean chroma (CIELAB a*, b* distance from 0) of ``img``."""
    lab = cv2.cvtColor(as_colour(img_bgr).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.mean(np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)))


def colour_cast(img_bgr: np.ndarray) -> dict:
    """Mean R/G and B/G of an image (1, 1 for a grey world)."""
    a = as_colour(img_bgr).reshape(-1, 3).astype(np.float64)
    m = a.mean(axis=0) + 1e-9
    return {"r_g": float(m[2] / m[1]), "b_g": float(m[0] / m[1])}
