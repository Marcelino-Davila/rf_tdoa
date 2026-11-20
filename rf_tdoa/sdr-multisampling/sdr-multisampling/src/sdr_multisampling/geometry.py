"""Physical bounds, cycle closure, and bounded multi-start 2D localization."""
import numpy as np
from scipy.optimize import least_squares

C = 299_792_458.0


def baseline_bound(a, b):
    return float(np.linalg.norm(np.asarray(a)-np.asarray(b)) / C)


def closure_errors(pair_delays):
    """Input (i,j): t_j-t_i; output independent triangle diagnostics."""
    nodes = sorted({n for pair in pair_delays for n in pair})
    out = {}
    for a in range(len(nodes)):
        for b in range(a+1, len(nodes)):
            for c in range(b+1, len(nodes)):
                i,j,k = nodes[a],nodes[b],nodes[c]
                if all(p in pair_delays for p in ((i,j),(j,k),(i,k))):
                    out[f"{i}|{j}|{k}"] = pair_delays[i,j]+pair_delays[j,k]-pair_delays[i,k]
    return out


def locate_2d(positions, delays, bounds, tolerance_m=.5):
    """Return candidate solutions within an explicit area of interest.

    No automatic arbitrary origin/triangle, no outside-region guarantee, no
    position covariance derived from correlated pair measurements.
    """
    names = sorted(positions)
    pts = np.array([positions[n] for n in names], float)
    if pts.shape != (len(names), 2) or len(names) < 3 or not np.isfinite(pts).all():
        raise ValueError("at least three finite 2D receiver positions required")
    if np.linalg.matrix_rank(pts-pts.mean(axis=0)) < 2:
        raise ValueError("collinear receiver geometry")
    lo, hi = np.asarray(bounds, float)
    if lo.shape != (2,) or hi.shape != (2,) or not np.isfinite([lo,hi]).all() or np.any(lo >= hi):
        raise ValueError("bounds must be [[xmin,ymin],[xmax,ymax]]")
    if tolerance_m <= 0 or not np.isfinite(tolerance_m):
        raise ValueError("positive residual tolerance required")
    if len(delays) < 2 or any(i not in positions or j not in positions for i,j in delays):
        raise ValueError("insufficient/unknown receiver pairs")
    edges = list(delays)
    incidence = np.zeros((len(edges), len(names)))
    for k,(i,j) in enumerate(edges):
        incidence[k,names.index(i)] = -1
        incidence[k,names.index(j)] = 1
        if not np.isfinite(delays[i,j]) or abs(delays[i,j]) > baseline_bound(positions[i],positions[j])+tolerance_m/C:
            return {"accepted": False, "reason": "nonphysical_delay", "candidates": []}
    if np.linalg.matrix_rank(incidence) < len(names)-1:
        return {"accepted": False, "reason": "disconnected_pair_graph", "candidates": []}
    target = np.array([delays[e]*C for e in edges])
    def residual(p):
        d = np.linalg.norm(pts-p,axis=1)
        return incidence @ d-target
    candidates = []
    for u in np.linspace(.05,.95,5):
        for v in np.linspace(.05,.95,5):
            start = lo+(hi-lo)*[u,v]
            fit = least_squares(residual,start,bounds=(lo,hi),loss="soft_l1",f_scale=tolerance_m,
                                xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=1000)
            rms = float(np.sqrt(np.mean(residual(fit.x)**2)))
            if not fit.success or rms > tolerance_m:
                continue
            if any(np.linalg.norm(fit.x-np.array(c["xy_m"])) < .01 for c in candidates):
                continue
            sv = np.linalg.svd(fit.jac,compute_uv=False)
            condition = float(sv[0]/max(sv[-1],1e-30))
            edge = bool(np.any(np.minimum(fit.x-lo,hi-fit.x) < .001))
            candidates.append({"xy_m":fit.x.tolist(),"residual_rms_m":rms,
                               "jacobian_condition":condition,"on_region_boundary":edge})
    candidates.sort(key=lambda c:c["residual_rms_m"])
    accepted = bool(len(candidates)==1 and not candidates[0]["on_region_boundary"]
                    and candidates[0]["jacobian_condition"] < 1000)
    return {"accepted":accepted,"reason":"ok" if accepted else "ambiguous_or_poor_fit",
            "candidates":candidates,"search_scope":"specified_bounds_only"}
