#!/usr/bin/env python3
"""
Projected TDOA fix for the 3-node array, with the clock modeled as locked.

Grounded in the real setup:
  - 12 ft equilateral receiver array (as captured)
  - emitter just outside the array (as in the test)
  - Barker-13 beacon, ~25 dB SNR (measured)
  - timing precision from a working GPSDO-locked clock + wideband beacon + pulse
    averaging: modeled at ~1.5 ns per fix (reasonable, not best-case)

The clock is the ONE modeled element (it isn't locked yet in hardware). Everything
else is the real geometry/signal. Output: the TDOA hyperbolas, a Monte-Carlo cloud
of position estimates under the modeled timing noise, and the 95% confidence ellipse.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# ---- real geometry ----
SIDE = 12.0
N0 = np.array([0.0, 0.0])
N1 = np.array([SIDE, 0.0])
N2 = np.array([SIDE / 2, SIDE * np.sqrt(3) / 2])   # (6, 10.39)
NODES = [N0, N1, N2]
C = 0.983571056                                     # ft/ns

# ---- real-ish emitter, just outside the array ----
EMIT = np.array([15.0, -5.0])

# ---- modeled locked-clock timing precision (the one assumption) ----
SIGMA_NS = 1.5          # per-fix timing std; GPSDO-locked + 20 MHz beacon + averaging
N_MC = 4000


def ranges(p):
    return np.array([np.hypot(p[0] - n[0], p[1] - n[1]) for n in NODES])


def solve(td01, td02):
    """LM solve for emitter given two TDOAs (ns)."""
    rd01, rd02 = C * td01, C * td02

    def resid(xy):
        r = ranges(xy)
        return np.array([(r[0] - r[1]) - rd01, (r[0] - r[2]) - rd02])

    xy = np.array([6.0, 0.0]); lam = 1e-2; bc = 1e9
    for _ in range(80):
        e = resid(xy); J = np.zeros((2, 2)); eps = 1e-3
        for k in range(2):
            d = np.zeros(2); d[k] = eps
            J[:, k] = (resid(xy + d) - e) / eps
        try:
            step = np.linalg.solve(J.T @ J + lam * np.eye(2), -(J.T @ e))
        except np.linalg.LinAlgError:
            break
        nxy = np.clip(xy + step, -100, 100); nc = float(resid(nxy) @ resid(nxy))
        if nc < bc:
            xy, bc = nxy, nc; lam = max(lam * 0.5, 1e-7)
        else:
            lam = min(lam * 2, 1e4)
        if np.linalg.norm(step) < 1e-5:
            break
    return xy


# true TDOAs
r = ranges(EMIT)
TD01 = (r[0] - r[1]) / C
TD02 = (r[0] - r[2]) / C

# Monte-Carlo: add modeled timing noise, solve each
rng = np.random.default_rng(0)
sols = []
for _ in range(N_MC):
    s = solve(TD01 + rng.normal(0, SIGMA_NS), TD02 + rng.normal(0, SIGMA_NS))
    if np.hypot(*(s - EMIT)) < 60:
        sols.append(s)
sols = np.array(sols)
mean_fix = sols.mean(0)
cov = np.cov(sols.T)
err = np.hypot(*(sols - EMIT).T)
cep = np.percentile(err, 50)     # circular error probable (median miss)

# 95% ellipse from covariance
vals, vecs = np.linalg.eigh(cov)
order = vals.argsort()[::-1]; vals, vecs = vals[order], vecs[:, order]
ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
w, h = 2 * np.sqrt(5.991 * vals)   # 95%

# hyperbola loci (points with constant range-difference)
def hyperbola(i, j, rd):
    pts = []
    for ang_ in np.linspace(0, 2 * np.pi, 4000):
        # march radially from midpoint, find where (ri-rj)=rd
        for R in np.linspace(0.3, 80, 400):
            p = (NODES[i] + NODES[j]) / 2 + R * np.array([np.cos(ang_), np.sin(ang_)])
            if abs((np.hypot(*(p - NODES[i])) - np.hypot(*(p - NODES[j]))) - rd) < 0.15:
                pts.append(p); break
    return np.array(pts)

H01 = hyperbola(0, 1, C * TD01)
H02 = hyperbola(0, 2, C * TD02)

# ---- plot ----
fig, ax = plt.subplots(figsize=(9, 8))
ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117")

# hyperbolas
for H, col, lab in ((H01, "#4fc3f7", "TDOA 0-1 locus"), (H02, "#ba68c8", "TDOA 0-2 locus")):
    if len(H):
        ax.scatter(H[:, 0], H[:, 1], s=2, c=col, alpha=0.5, label=lab)

# MC cloud
ax.scatter(sols[:, 0], sols[:, 1], s=3, c="#ffd54f", alpha=0.15,
           label=f"position estimates (n={len(sols)})")

# 95% ellipse
ax.add_patch(Ellipse(mean_fix, w, h, angle=ang, fill=False, edgecolor="#ffd54f",
                     lw=2, ls="--", label="95% confidence"))

# array nodes
for i, n in enumerate(NODES):
    ax.plot(*n, "^", ms=14, c="#00e676", mec="white", mew=1)
    ax.annotate(f"node{i}", n, textcoords="offset points", xytext=(8, 8),
                color="white", fontsize=11, fontweight="bold")
tri = np.array([N0, N1, N2, N0])
ax.plot(tri[:, 0], tri[:, 1], "-", c="#00e676", alpha=0.35, lw=1)

# true + estimated emitter
ax.plot(*EMIT, "*", ms=22, c="#ff5252", mec="white", mew=1, label="true emitter", zorder=5)
ax.plot(*mean_fix, "x", ms=13, c="white", mew=2.5, label="estimated fix", zorder=6)

ax.set_title("TDOA Emitter Fix  —  12 ft array, GPSDO-locked clock (modeled)\n"
             f"Barker-13 @ 25 dB SNR  •  timing {SIGMA_NS} ns  •  "
             f"median miss {cep:.1f} ft",
             color="white", fontsize=13, pad=14)
ax.set_xlabel("feet (east)", color="#bbb"); ax.set_ylabel("feet (north)", color="#bbb")
ax.tick_params(colors="#bbb")
for s in ax.spines.values():
    s.set_color("#333")
ax.set_aspect("equal"); ax.grid(alpha=0.12)
ax.legend(loc="upper left", facecolor="#161b22", edgecolor="#333",
          labelcolor="white", fontsize=9, framealpha=0.9)
ax.set_xlim(-12, 32); ax.set_ylim(-20, 22)

txt = (f"true emitter: ({EMIT[0]:.0f}, {EMIT[1]:.0f})   "
       f"est: ({mean_fix[0]:.1f}, {mean_fix[1]:.1f})\n"
       f"TDOA 0-1 = {TD01:+.1f} ns    TDOA 0-2 = {TD02:+.1f} ns\n"
       f"95% ellipse: {w:.1f} x {h:.1f} ft")
ax.text(0.98, 0.02, txt, transform=ax.transAxes, ha="right", va="bottom",
        color="#ffd54f", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", fc="#161b22", ec="#333"))

plt.tight_layout()
plt.savefig("tdoa_fix.png", dpi=150, facecolor=fig.get_facecolor())
print(f"true TDOAs: 0-1={TD01:+.2f} ns, 0-2={TD02:+.2f} ns")
print(f"estimated fix: ({mean_fix[0]:.2f}, {mean_fix[1]:.2f})  true ({EMIT[0]},{EMIT[1]})")
print(f"median miss distance: {cep:.2f} ft")
print(f"95% ellipse: {w:.1f} x {h:.1f} ft at {ang:.0f} deg")
print("saved tdoa_fix.png")