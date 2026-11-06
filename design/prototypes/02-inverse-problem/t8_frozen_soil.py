"""Are the three frozen-soil parameters live anywhere in the prior? (They are inert at the reference.)

At 30 prior draws with each frozen-soil parameter at its template value, moves each to two
far-apart values and counts the draws at which the features change.
"""

import numpy as np

from prior_table import draw, to_sipnet
from reference import features_batch

TEMPLATE = {"frozen_soil_threshold": 0.0, "frozen_soil_foliar_respiration_factor": 0.0, "frozen_soil_water_fraction": 1.0}
MOVES = {"frozen_soil_threshold": (-5.0, 5.0), "frozen_soil_foliar_respiration_factor": (0.5, 1.0), "frozen_soil_water_fraction": (0.0, 0.5)}
counts = dict.fromkeys(MOVES, 0)
for theta in draw(30, seed=21):
    base = to_sipnet(theta | TEMPLATE)
    points = [base] + [base | {name: value} for name, values in MOVES.items() for value in values]
    runs = features_batch(points)
    for k, name in enumerate(MOVES):
        moved = runs[1 + 2 * k : 3 + 2 * k]
        counts[name] += int(any(not np.array_equal(run, runs[0]) for run in moved))
print("draws (of 30) at which the parameter moved the features:", counts)
