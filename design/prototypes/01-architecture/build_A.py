import sys, pickle, numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, sys.argv[2]); import hf_operators as ops
steps = pd.read_pickle(sys.argv[1])
print("steps:", steps.shape, list(steps.columns)[:12], steps.index[0], steps.index[-1])
op = ops.build("halfmonth_day_night", steps)
f = op.features
print("features:", op.size, f["class"].value_counts().to_dict(), "| A:", op.matrix.shape)
y = op.apply(steps["observed"].to_numpy())
sp.save_npz(sys.argv[3] + "/A.npz", op.matrix)
f.assign(y=y).to_csv(sys.argv[3] + "/features.csv", index=False)
print(f.assign(y=y).groupby("class")["y"].describe().round(2))
