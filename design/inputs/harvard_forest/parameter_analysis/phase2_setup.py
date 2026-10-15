"""Phase 2 setup: build the calibration vector once and cache what the other scripts need.

Writes phase2/cache.pkl: the FastPredictor, y, per-observation source name,
time label, total noise sd (sqrt diag of each block), and the IC ensemble
quantiles. Also checks fast == library on one base run.
"""
import sys, pickle, warnings
from pathlib import Path
warnings.simplefilter("ignore")
HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
sys.path.insert(0, str(EXP)); sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd

def main():
    import config, runs, observations, noise, fast_forward, inputs
    import phase2_common as pc
    vec = observations.calibration_observation_vector()
    blocks = noise.noise_covariance_blocks(vec, config.CALIBRATION_NEE_SERIES)
    sd = np.concatenate([np.sqrt(np.diag(b)) for b in blocks.values()])
    idx = vec.index
    src = np.asarray(idx.get_level_values("observation_source"))
    t = pd.DatetimeIndex(idx.get_level_values("time"))
    y = np.asarray(vec.flat(vec.fields(np.zeros(vec.dimension))))  # placeholder, replaced below
    y = np.asarray(vec.y) if hasattr(vec, "y") else None
    base = pc.overrides({})
    res = runs.sipnet_model()(**base)
    ends = pd.DatetimeIndex(res.outputs.xarray["time"].values)
    pred = fast_forward.FastPredictor.from_vector(vec, ends)
    print("fast vs library max abs diff:", fast_forward.check_fast_predictions(vec, res, base, pred))
    icf = inputs.initial_condition_fields()
    ic = {}
    for n, f in icf.items():
        v = np.asarray(f.squeeze()).ravel()
        ic[n] = np.quantile(v, [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975])
        print(n, np.round(ic[n], 2))
    # the likelihood's full block structure for final scoring
    with open(HERE / "phase2" / "cache.pkl", "wb") as fh:
        pickle.dump(dict(predictor=pred, y=y, sd=sd, source=src, time=t, ic=ic,
                         blocks={k: np.asarray(b) for k, b in blocks.items()},
                         source_order=list(blocks)), fh)
    print("N", len(sd), {s: int((src == s).sum()) for s in blocks})
    for s in blocks:
        m = src == s
        print(s, "y range", np.nanmin(y[m]).round(2), np.nanmax(y[m]).round(2), "sd median", np.median(sd[m]).round(2))

if __name__ == "__main__":
    main()
