"""Bounded-memory replay pipeline; no hardware dependencies or invented clocks."""
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
import hashlib
import json
import numpy as np
from . import __version__
from .sampling import Clock, IQFile, aligned_window
from .filters import Band, matched_filter, detection_score
from .delay import estimate_delay, robust_fuse
from .geometry import C, baseline_bound, closure_errors, locate_2d


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def run(config, base_dir="."):
    """Analyze a manifest; all file paths are relative to that manifest.

    clock.epoch_s refers to ADC sample time; hardware_delay_s is extra analog/
    digital signal-path delay. They are different calibrations; do not subtract
    the same offset twice. Frequency correction is configured, not inferred.
    """
    root = Path(base_dir)
    fs = _positive(config["sample_rate_hz"], "sample rate")
    size = config.get("window_samples", 32768)
    windows = config.get("windows", 16)
    if not isinstance(size,int) or size < 256 or not isinstance(windows,int) or not 1 <= windows <= 10000:
        raise ValueError("window_samples >=256 and windows in 1..10000 required")
    start_sample = config.get("start_sample", 256)
    if not isinstance(start_sample,int) or start_sample < 0:
        raise ValueError("start_sample must be a nonnegative integer")
    max_delay = _positive(config.get("max_delay_s", 5e-6), "search radius")
    timing_budget = _positive(config.get("timing_budget_s", 20e-9), "timing budget")
    agreement = _positive(config.get("band_agreement_s", 20e-9), "band agreement")
    closure_limit = _positive(config.get("closure_limit_s", 20e-9), "closure limit")
    minimum = config.get("minimum_windows", 3)
    if not isinstance(minimum,int) or not 1 <= minimum <= windows:
        raise ValueError("minimum_windows outside requested window count")
    bands = [Band(**b) for b in config["bands"]]
    if not bands or len({b.name for b in bands}) != len(bands):
        raise ValueError("unique nonempty bands required")
    for b in bands:
        b.coefficients(fs)
        if (size-b.taps+1)//b.decimation < 32:
            raise ValueError("too few samples after channelization")
    receivers = config["receivers"]
    if len(receivers) < 2 or len({r["name"] for r in receivers}) != len(receivers):
        raise ValueError("at least two unique receivers required")
    sources, metadata, taps_by_rx = {}, {}, {}
    for r in receivers:
        name = r["name"]
        clock = Clock(**r["clock"])
        source = IQFile(root/r["path"], clock, r.get("format","cf32_le"),r.get("invalid_ranges",[]))
        for field in ("hardware_delay_s","frequency_correction_hz"):
            if not np.isfinite(float(r.get(field,0))):
                raise ValueError(f"nonfinite {field}")
        if "timing_uncertainty_s" in r and (not np.isfinite(r["timing_uncertainty_s"]) or r["timing_uncertainty_s"] < 0):
            raise ValueError("invalid timing uncertainty")
        sources[name], metadata[name] = source,r
        if "template_path" in r:
            s = np.load(root/r["template_path"],allow_pickle=False)
            covariance = np.load(root/r["noise_covariance_path"],allow_pickle=False) if "noise_covariance_path" in r else None
            taps_by_rx[name] = matched_filter(s,covariance)
    names = sorted(sources)
    pairs = list(combinations(names,2))
    positions = {r["name"]:r["position_m"] for r in receivers if "position_m" in r}
    if positions and (len(positions)!=len(receivers) or np.asarray(list(positions.values())).shape!=(len(receivers),2)
                      or not np.isfinite(list(positions.values())).all()):
        raise ValueError("provide finite 2D geometry for every receiver or none")
    calibrated = all(r.get("timing_verified") is True and r.get("calibration_id")
                     and "timing_uncertainty_s" in r and "hardware_delay_s" in r for r in receivers)
    uncertainty = {p: metadata[p[0]].get("timing_uncertainty_s",0)+metadata[p[1]].get("timing_uncertainty_s",0)
                   for p in pairs}
    calibrated = bool(calibrated and all(v <= timing_budget for v in uncertainty.values()))
    records, window_records = [],[]
    accepted_by_pair = {p:[] for p in pairs}
    options = config.get("estimator",{})
    allowed = {"interpolation","phat_exponent","min_correlation","min_pslr_db","min_rms_bandwidth_hz"}
    if set(options)-allowed:
        raise ValueError(f"unknown estimator settings: {set(options)-allowed}")
    for w in range(windows):
        first = start_sample+w*size  # strictly nonoverlapping windows
        raw, errors, detection = {},{},{}
        for name in names:
            try:
                x = aligned_window(sources[name],first/fs,size,fs)
                correction = float(metadata[name].get("frequency_correction_hz",0))
                if correction:
                    x *= np.exp(-2j*np.pi*correction*(first+np.arange(size))/fs)
                raw[name] = x
                if name in taps_by_rx:
                    score = detection_score(x,taps_by_rx[name])
                    detection[name] = score
                    if score < config.get("minimum_detection_score_db",8):
                        raise ValueError("matched-filter detection gate failed")
            except ValueError as exc:
                raw.pop(name,None)
                errors[name] = str(exc)
        window_records.append({"window":w,"start_sample":first,"errors":errors,
                               "detection_peak_to_median_db":detection})
        band_results = {p:[] for p in pairs}
        for b in bands:
            branch = {n:b.apply(x,fs,first)[0] for n,x in raw.items()}
            branch_fs = fs/b.decimation
            for i,j in pairs:
                row = {"window":w,"band":b.name,"pair":[i,j],"start_sample":first}
                if i not in branch or j not in branch:
                    row.update(accepted=False,reasons=["missing_or_invalid_receiver_window"])
                    records.append(row)
                    continue
                try:
                    result = estimate_delay(branch[i],branch[j],branch_fs,max_delay,**options)
                    row.update(result.to_dict())
                    bias = float(metadata[j].get("hardware_delay_s",0)-metadata[i].get("hardware_delay_s",0))
                    row["corrected_delay_s"] = result.delay_s-bias
                    row["range_difference_m"] = row["corrected_delay_s"]*C
                    if calibrated and positions and abs(row["corrected_delay_s"]) > baseline_bound(positions[i],positions[j])+uncertainty[i,j]:
                        row["accepted"] = False
                        row["reasons"].append("outside_physical_baseline")
                    if row["accepted"]:
                        band_results[i,j].append(row["corrected_delay_s"])
                except ValueError as exc:
                    row.update(accepted=False,reasons=[str(exc)])
                records.append(row)
        # Bands are correlated evidence: one vote per time window, no sqrt(B) gain.
        per_window = {}
        for pair,values in band_results.items():
            if values and np.ptp(values) <= agreement:
                per_window[pair] = float(np.median(values))
            elif values:
                window_records[-1].setdefault("band_disagreement",[]).append(list(pair))
        closure = closure_errors(per_window)
        window_records[-1]["closure_s"] = closure
        if any(abs(v)>closure_limit for v in closure.values()):
            window_records[-1]["rejected"] = "pair_cycle_closure"
            continue
        for pair,value in per_window.items():
            accepted_by_pair[pair].append(value)
    aggregates = {p:robust_fuse(v,minimum,1/(fs*options.get("interpolation",32)))
                  for p,v in accepted_by_pair.items()}
    for aggregate in aggregates.values():
        if aggregate.get("repeatability_mad_s", 0) > timing_budget:
            aggregate["accepted"] = False
            aggregate["reason"] = "delay_varies_beyond_timing_budget"
    delays = {p:a["delay_s"] for p,a in aggregates.items() if a["accepted"]}
    final_closure = closure_errors(delays)
    location = {"accepted":False,"reason":"requires_verified_timing_geometry_and_region"}
    if calibrated and positions and "localization_bounds_m" in config:
        if len(delays)!=len(pairs):
            location = {"accepted":False,"reason":"missing_accepted_pairs"}
        elif any(abs(v)>closure_limit for v in final_closure.values()):
            location = {"accepted":False,"reason":"aggregate_pair_cycle_closure"}
        else:
            location = locate_2d(positions,delays,config["localization_bounds_m"],
                                 config.get("position_residual_tolerance_m",.5))
    manifest_hash = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    return {"schema_version":1,"software_version":__version__,"config_sha256":manifest_hash,
            "config":config,"timing_verified_for_localization":calibrated,
            "input_files":{n:{"path":str(s.path),"samples":s.count,"bytes":s.path.stat().st_size}
                           for n,s in sources.items()},
            "interpretation":"calibrated propagation-delay estimates" if calibrated else "diagnostic stream offsets; NOT verified propagation TDOA",
            "bands":[asdict(b) for b in bands],"windows":window_records,"measurements":records,
            "pairs":{"|".join(p):a for p,a in aggregates.items()},
            "closure_s":final_closure,"location":location}
