import argparse
import json
from pathlib import Path
import numpy as np
from scipy import signal
from .pipeline import run
from .geometry import C


def simulate(directory):
    """Explicitly synthetic, seeded wideband 3-RX fixture with fractional delays."""
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(20260928)
    fs, n = 20e6, 65536
    positions = [[0.,0.],[12.,0.],[4.,10.]]
    source = np.array([5.,4.])
    distance = np.linalg.norm(np.array(positions)-source,axis=1)
    delays = (distance-distance[0])/C
    h = signal.firwin(97,3e6,fs=fs)
    excitation = signal.fftconvolve(rng.normal(size=n+512)+1j*rng.normal(size=n+512),h,mode="same")
    # Independent time-domain windowed-sinc synthesis (not the estimator's DFT).
    offsets = np.arange(-64,65)
    receivers=[]
    for k,tau in enumerate(delays):
        d = tau*fs
        kernel = np.sinc(offsets-d)*np.kaiser(len(offsets),10)
        kernel /= kernel.sum()
        x = signal.fftconvolve(excitation,kernel,mode="same")[256:256+n]
        x *= np.exp(1j*[0.,1.7,-2.2][k])
        x += .015*(rng.normal(size=n)+1j*rng.normal(size=n))
        name=f"rx{k}"
        x.astype("<c8").tofile(directory/f"{name}.cfile")
        receivers.append({"name":name,"path":f"{name}.cfile","format":"cf32_le",
                          "clock":{"rate_hz":fs,"epoch_s":0},"position_m":positions[k],
                          "hardware_delay_s":0,"timing_verified":True,"timing_uncertainty_s":1e-10,
                          "calibration_id":"SYNTHETIC-ONLY"})
    config={"sample_rate_hz":fs,"window_samples":4096,"windows":12,"start_sample":256,
            "max_delay_s":200e-9,"minimum_windows":4,"band_agreement_s":10e-9,
            "bands":[{"name":"wide","center_hz":0,"bandwidth_hz":6e6,"taps":129},
                     {"name":"decimated","center_hz":0,"bandwidth_hz":5e6,"decimation":2,"taps":129}],
            "receivers":receivers,"localization_bounds_m":[[-10,-10],[25,25]],
            "position_residual_tolerance_m":.25,
            "synthetic_truth":{"source_m":source.tolist(),"delays_relative_rx0_s":delays.tolist()}}
    path=directory/"manifest.json"
    path.write_text(json.dumps(config,indent=2)+"\n")
    return path


def main(argv=None):
    p=argparse.ArgumentParser(description="Multi-receiver IQ replay, filtering and calibrated TDOA")
    sub=p.add_subparsers(dest="command",required=True)
    analyze=sub.add_parser("analyze")
    analyze.add_argument("manifest",type=Path)
    analyze.add_argument("--output",type=Path,required=True)
    demo=sub.add_parser("demo",help="generate and analyze explicitly SYNTHETIC data")
    demo.add_argument("directory",type=Path)
    args=p.parse_args(argv)
    if args.command=="demo":
        path=simulate(args.directory)
        output=args.directory/"result.json"
    else:
        path,output=args.manifest,args.output
    config=json.loads(path.read_text())
    report=run(config,path.parent)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(report["interpretation"])
    for name,result in report["pairs"].items():
        print(name,json.dumps(result))
    print("location:",json.dumps(report["location"]))
    print("report:",output)


if __name__=="__main__":
    main()
