#!/usr/bin/env python
"""Loading and track math for the marginalization npz files written by marginalize_tracks.py.

Track construction (see README): average the profile in LOG space, renormalize, then scale by
exp(mean log-counts). Averaging linearly instead lets the 2-3 highest-count backgrounds dominate
and breaks the identity  area(T_motif) / area(T_bg) == exp(delta_logcounts).
"""
import json

import numpy as np

OUT_LEN = 1000
CENTER = OUT_LEN // 2


def load(path):
    """-> (conditions, meta). conditions[bgset][name] = {'logq': (nbg,1000), 'c': (nbg,)}."""
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    cond = {}
    for k in z.files:
        if k == "meta":
            continue
        if k.endswith("__meta"):
            bgset, name = k.split("::", 1)
            cond.setdefault(bgset, {}).setdefault(name[:-6], {})["meta"] = json.loads(str(z[k]))
            continue
        bgset, rest = k.split("::", 1)
        name, field = rest.rsplit("::", 1)
        cond.setdefault(bgset, {}).setdefault(name, {})[field] = z[k]
    return cond, meta


def log_softmax(v):
    v = v - v.max()
    return v - np.log(np.exp(v).sum())


def softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()


def shape_logq(cond):
    """Geometric-mean profile shape of one condition, renormalized. -> log q, (1000,)."""
    return log_softmax(cond["logq"].astype(np.float64).mean(0))


def track(cond):
    """Counts-scaled accessibility track: q(x) * exp(mean log-counts). area == exp(mean c)."""
    return np.exp(shape_logq(cond)) * np.exp(cond["c"].mean())


def dlog(cond, bg):
    """Marginalization effect in log-counts: mean(c_condition) - mean(c_background)."""
    return float(cond["c"].mean() - bg["c"].mean())


def additive_track(cA, cB, bg, clip=10.0):
    """Log-additive expectation, shape and area decoupled.

    The naive y_A*y_B/y_bg has area exp(dA+dB)*area_bg*sum_x qA qB/qbg, and that correction
    factor exceeds 1 exactly when both inserts concentrate signal centrally -- which would push
    the additive track up and visually shrink the synergy gap. Renormalizing the shape and
    applying the area separately makes area(T_add)/area(T_bg) == exp(dA+dB) exactly.
    """
    u = shape_logq(cA) + shape_logq(cB) - shape_logq(bg)
    u = np.clip(u, np.median(u) - clip, np.median(u) + clip)
    return softmax(u) * np.exp(cA["c"].mean() + cB["c"].mean() - bg["c"].mean())


def log_ratio_track(cond, bg):
    """Per-base log enrichment over background: log q_c + c_bar_c - log q_bg - c_bar_bg.

    Additivity is literal addition in this space, so no renormalization argument is needed.
    """
    return (shape_logq(cond) + cond["c"].mean()) - (shape_logq(bg) + bg["c"].mean())


def window(arr, half=150, center=CENTER):
    return arr[center - half:center + half]
