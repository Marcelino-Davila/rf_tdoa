import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy import signal
from sdr_multisampling.filters import Band, StreamingFIR, noise_covariance, matched_filter
from sdr_multisampling.sampling import Clock, IQFile, aligned_window, fit_clock, ContiguousBuffer, MARKER_SYNC
from sdr_multisampling.delay import estimate_delay, robust_fuse
from sdr_multisampling.geometry import C, closure_errors, locate_2d
from sdr_multisampling.calibration import hardware_offsets
from sdr_multisampling.cli import simulate
from sdr_multisampling.pipeline import run


def waveform(n=4096):
    rng=np.random.default_rng(3)
    x=rng.normal(size=n+512)+1j*rng.normal(size=n+512)
    return signal.fftconvolve(x,signal.firwin(81,.45),mode="same")


def shifted(x,d):
    n=np.arange(-80,81)
    h=np.sinc(n-d)*np.kaiser(len(n),10)
    return signal.fftconvolve(x,h/h.sum(),mode="same")[256:-256]


class DelayTests(unittest.TestCase):
    def test_fractional_delays_and_unknown_phase(self):
        x=waveform()
        for d in (-7.31,-.37,0,.41,6.83):
            with self.subTest(d=d):
                r=estimate_delay(x[256:-256],shifted(x,d)*np.exp(2.3j),20e6,1e-6)
                self.assertTrue(r.accepted,r.reasons)
                self.assertAlmostEqual(r.delay_samples,d,delta=.015)

    def test_sign_against_scipy(self):
        x=waveform()[256:-256]
        y=np.r_[np.zeros(9),x[:-9]]
        lag=signal.correlation_lags(len(y),len(x))[np.argmax(abs(signal.correlate(y,x)))]
        r=estimate_delay(x,y,20e6,1e-6,phat_exponent=0)
        self.assertEqual(lag,9)
        self.assertAlmostEqual(r.delay_samples,lag,delta=.02)

    def test_bounded_search_and_bad_signals(self):
        x=waveform()
        r=estimate_delay(x[256:-256],shifted(x,5),20e6,1/20e6)
        self.assertFalse(r.accepted)
        n=np.arange(4096)
        tone=np.exp(2j*np.pi*.1*n)
        self.assertFalse(estimate_delay(tone,tone*np.exp(1j),20e6,1e-6).accepted)
        rng=np.random.default_rng(9)
        self.assertFalse(estimate_delay(rng.normal(size=4096),rng.normal(size=4096),20e6,1e-6).accepted)
        with self.assertRaises(ValueError):
            estimate_delay(np.zeros(512),np.zeros(512),20e6,1e-6)

    def test_subsample_physical_search(self):
        x=waveform()
        r=estimate_delay(x[256:-256],shifted(x,.3),20e6,.5/20e6)
        self.assertTrue(r.accepted,r.reasons)
        self.assertAlmostEqual(r.delay_samples,.3,delta=.015)

    def test_periodic_ambiguity(self):
        rng=np.random.default_rng(41)
        x=np.tile(rng.normal(size=32)+1j*rng.normal(size=32),128)
        r=estimate_delay(x,np.roll(x,3),20e6,5/20e6)
        self.assertFalse(r.accepted)
        self.assertIn("ambiguous_correlation_peak",r.reasons)

    def test_fusion(self):
        r=robust_fuse([1e-8,1.01e-8,.99e-8,1e-8,1e-6])
        self.assertTrue(r["accepted"])
        self.assertEqual(r["rejected_outliers"],1)
        self.assertAlmostEqual(r["delay_s"],1e-8)


class FilterTests(unittest.TestCase):
    def test_streaming_matches_convolution_including_tail(self):
        rng=np.random.default_rng(7)
        for length in (1,17,137):
            x=rng.normal(size=1057)+1j*rng.normal(size=1057)
            h=rng.normal(size=length)+1j*rng.normal(size=length)
            f=StreamingFIR(h,256)
            chunks=[f.feed(x[a:b]) for a,b in ((0,1),(1,17),(17,813),(813,1057))]
            chunks.append(f.flush_tail())
            np.testing.assert_allclose(np.concatenate(chunks),np.convolve(x,h),atol=1e-12)

    def test_covariance_complex_orientation_and_filter(self):
        n=np.arange(4096)
        R=noise_covariance(np.exp(.3j*n),8)
        self.assertGreater(R[1,0].imag,0)
        template=np.array([1+2j,3-1j,-2+.3j])
        R=np.array([[2,.3j,0],[-.3j,1,0],[0,0,3]],complex)
        h=matched_filter(template,R,loading=0)
        self.assertAlmostEqual(abs(np.convolve(template,h)[len(template)-1]),1,places=12)
        g=h[::-1].conj()
        optimum=np.vdot(template,np.linalg.solve(R,template)).real
        snr=abs(np.vdot(g,template))**2/np.vdot(g,R@g).real
        self.assertAlmostEqual(snr,optimum,places=10)

    def test_decimation_delay_and_antialias_validation(self):
        x=waveform()
        b=Band("test",0,6e6,2,129)
        a,fs,_=b.apply(x[256:-256],20e6)
        y,_,_=b.apply(shifted(x,1.7),20e6)
        r=estimate_delay(a,y,fs,1e-6)
        self.assertAlmostEqual(r.delay_s,1.7/20e6,delta=1e-9)
        with self.assertRaises(ValueError):Band("bad",0,1.5e6,10,9).coefficients(20e6)


class SamplingTests(unittest.TestCase):
    def test_clock_fit_and_jitter_rejection(self):
        rate=20e6*(1+3e-6)
        n=np.arange(10)*rate
        c,r=fit_clock(n,np.arange(10)+.01,20e6,1e-8)
        self.assertAlmostEqual(c.rate_hz,rate,places=3)
        self.assertAlmostEqual(c.epoch_s,.01,places=10)
        with self.assertRaises(ValueError):fit_clock(n,np.arange(10)+np.arange(10)%2*.001,20e6,1e-6)

    def test_clock_resampling_against_analytic_signal(self):
        with tempfile.TemporaryDirectory() as td:
            fs=20e6
            clock=Clock(fs*(1+30e-6),.3/fs)
            n=np.arange(8000)
            t=clock.epoch_s+n/clock.rate_hz
            x=np.exp(2j*np.pi*1.7e6*t)+.3*np.exp(-2j*np.pi*3.1e6*t)
            path=Path(td)/"rx.cfile"
            x.astype("<c8").tofile(path)
            a=aligned_window(IQFile(path,clock),100/fs,4096,fs)
            t=(100+np.arange(4096))/fs
            truth=np.exp(2j*np.pi*1.7e6*t)+.3*np.exp(-2j*np.pi*3.1e6*t)
            np.testing.assert_allclose(a,truth,atol=1e-4)

    def test_gaps_and_markers(self):
        b=ContiguousBuffer(20)
        b.feed(100,np.ones(10))
        with self.assertRaises(ValueError):b.feed(111,np.ones(2))
        with self.assertRaises(BufferError):b.feed(110,np.ones(11))
        self.assertEqual(b.pop(7)[0],100)
        b.feed(110,np.ones(3))
        self.assertEqual(b.pop(6)[0],107)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"x.iq"
            path.write_bytes(bytes(50)+MARKER_SYNC+bytes(4)+bytes(100))
            with self.assertRaises(ValueError):IQFile(path,Clock(20e6),"ci8").read(20,30)
            with self.assertRaises(ValueError):IQFile(path,Clock(20e6),"ci8",[(3,8)]).read(0,5)


class GeometryTests(unittest.TestCase):
    def test_known_position_and_closure(self):
        p={"a":[0,0],"b":[12,0],"c":[4,10],"d":[15,15]}
        truth=np.array([5,4])
        from itertools import combinations
        delays={(i,j):(np.linalg.norm(truth-p[j])-np.linalg.norm(truth-p[i]))/C for i,j in combinations(p,2)}
        self.assertLess(max(abs(v) for v in closure_errors(delays).values()),1e-20)
        r=locate_2d(p,delays,[[-10,-10],[25,25]],.01)
        self.assertTrue(r["accepted"],r)
        np.testing.assert_allclose(r["candidates"][0]["xy_m"],truth,atol=1e-5)
        delays["a","b"]=1
        self.assertFalse(locate_2d(p,delays,[[-10,-10],[25,25]])["accepted"])

    def test_calibration_keeps_known_propagation(self):
        r=hardware_offsets({"b":[10/C+2e-8]*6},{"a":[0,0],"b":[10,0]},[0,0],"a")
        self.assertAlmostEqual(r["b"]["hardware_delay_s"],2e-8)


class PipelineTests(unittest.TestCase):
    def test_end_to_end_and_unverified_gate(self):
        with tempfile.TemporaryDirectory() as td:
            path=simulate(td)
            config=json.loads(path.read_text())
            config["windows"]=4
            config["minimum_windows"]=3
            result=run(config,td)
            truth=config["synthetic_truth"]["delays_relative_rx0_s"]
            for name,k in (("rx0|rx1",1),("rx0|rx2",2)):
                self.assertTrue(result["pairs"][name]["accepted"])
                self.assertAlmostEqual(result["pairs"][name]["delay_s"],truth[k],delta=1e-9)
            self.assertTrue(result["location"]["accepted"],result["location"])
            config["receivers"][0]["timing_verified"]=False
            result=run(config,td)
            self.assertFalse(result["location"]["accepted"])
            self.assertFalse(result["timing_verified_for_localization"])

    def test_missing_data_rejected_without_position(self):
        with tempfile.TemporaryDirectory() as td:
            config=json.loads(simulate(td).read_text())
            config["windows"]=4
            config["receivers"][1]["invalid_ranges"]=[[0,65000]]
            result=run(config,td)
            self.assertFalse(result["pairs"]["rx0|rx1"]["accepted"])
            self.assertFalse(result["location"]["accepted"])



class CalibrationIntegrationTests(unittest.TestCase):
    def test_clock_epoch_drift_cfo_and_hardware_bias_together(self):
        with tempfile.TemporaryDirectory() as td:
            rng=np.random.default_rng(190)
            fs=20e6
            frequencies=rng.uniform(-3e6,3e6,80)
            amplitudes=np.exp(2j*np.pi*rng.random(80))/np.sqrt(80)
            receivers=[]
            propagation=[0,-.37/fs]
            biases=[0,70e-9]
            for k in range(2):
                rate=fs*(1+k*25e-6)
                epoch=k*.3/fs
                cfo=k*1700
                t=epoch+np.arange(14000)/rate
                x=np.exp(2j*np.pi*(t[:,None]-propagation[k]-biases[k])*frequencies)@amplitudes
                x*=np.exp(2j*np.pi*cfo*t)
                path=Path(td)/f'rx{k}.cfile'
                x.astype('<c8').tofile(path)
                receivers.append({'name':f'rx{k}','path':path.name,'clock':{'rate_hz':rate,'epoch_s':epoch},
                                  'hardware_delay_s':biases[k],'frequency_correction_hz':cfo,
                                  'timing_verified':True,'calibration_id':'test','timing_uncertainty_s':1e-10})
            config={'sample_rate_hz':fs,'window_samples':2048,'windows':5,'minimum_windows':3,
                    'max_delay_s':1e-6,'start_sample':100,'receivers':receivers,
                    'bands':[{'name':'wide','center_hz':0,'bandwidth_hz':6e6,'taps':129}]}
            report=run(config,td)
            estimate=report['pairs']['rx0|rx1']
            self.assertTrue(estimate['accepted'],estimate)
            self.assertAlmostEqual(estimate['delay_s'],propagation[1],delta=1e-9)

if __name__=="__main__":unittest.main()
