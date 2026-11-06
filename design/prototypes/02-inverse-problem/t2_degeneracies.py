"""Check the exact degeneracies of SIPNET's NEE features at the reference point.

Each direction scales some parameters by k and others by 1/k. It is exact when the features at
k = 1.3 and k = 0.7 equal the reference features to round-off. A non-direction is checked alongside
as a control.
"""

import numpy as np

from reference import features, features_batch, reference_point

base = reference_point()
reference = features(base)

DIRECTIONS = {
    "aMax*k, cFracLeaf*k": {"max_photosynthesis_rate": 1, "leaf_carbon_fraction": 1},
    "aMax*k, aMaxFrac/k, baseFolRespFrac/k": {
        "max_photosynthesis_rate": 1,
        "daily_mean_photosynthesis_fraction": -1,
        "foliar_respiration_fraction": -1,
    },
    "leafCSpWt*k, attenuation*k (laiInit = 0)": {"leaf_carbon_per_area": 1, "light_extinction_coefficient": 1},
    "control: aMax*k alone": {"max_photosynthesis_rate": 1},
    "control: leafCSpWt*k alone": {"leaf_carbon_per_area": 1},
}

points, labels = [], []
for label, powers in DIRECTIONS.items():
    for k in (1.03, 0.97):
        points.append(base | {name: base[name] * k**power for name, power in powers.items()})
        labels.append((label, k))
runs = features_batch(points)
scale = np.max(np.abs(reference))
for (label, k), run in zip(labels, runs):
    relative = np.max(np.abs(run - reference)) / scale
    print(f"{label:45s} k={k}: max relative change {relative:.2e}")
