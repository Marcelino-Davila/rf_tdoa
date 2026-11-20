"""Calibration from known-source runs; never use an unknown target to zero TDOA."""
import numpy as np
from .geometry import C
from .delay import robust_fuse


def hardware_offsets(measured_delays, positions, source_position, reference, minimum=5):
    """measured_delays[name] = repeated t_name-t_reference after clock alignment.

    Returns relative signal-path bias and empirical scatter. Splitter/cable test:
    supply equal-path measurements and corresponding zero geometric differences
    (e.g. colocated virtual positions), then separately account for cable lengths.
    Caller must add clock/PPS/cable uncertainties and record calibration provenance.
    """
    if reference not in positions or reference in measured_delays:
        raise ValueError("reference must have a position and no relative-delay series")
    p = np.asarray(source_position,float)
    coords = {k:np.asarray(v,float) for k,v in positions.items()}
    if not np.isfinite(p).all() or any(v.shape!=p.shape or not np.isfinite(v).all() for v in coords.values()):
        raise ValueError("invalid calibration geometry")
    d0 = np.linalg.norm(p-coords[reference])
    result = {reference:{"hardware_delay_s":0.,"repeatability_mad_s":0.}}
    for name,values in measured_delays.items():
        if name not in positions:
            raise ValueError("unknown receiver")
        f = robust_fuse(values,minimum)
        if not f["accepted"]:
            raise ValueError(f"insufficient calibration measurements for {name}")
        propagation = (np.linalg.norm(p-coords[name])-d0)/C
        result[name] = {"hardware_delay_s":float(f["delay_s"]-propagation),
                        "repeatability_mad_s":f["repeatability_mad_s"]}
    return result
