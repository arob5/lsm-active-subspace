"""Add the Morris screening score (max relative mu* over outputs) and the reason for fixing to fixed_parameters.csv."""
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
f = pd.read_csv(HERE / "fixed_parameters.csv")
score = pd.read_csv(HERE / "phase2" / "morris_relative_mu_star.csv", index_col=0)["score"]
f["morris_score"] = f.flat_name.map(score).round(2)
why = {
    "light_extinction_coefficient": "Screened 0.49 (day NEE only). Fixed at BETY median: flux-side it acts as attenuation/leafCSpWt with leaf carbon (phase 1 3.2), which leafGrowth/allocation already move.",
    "water_removal_fraction": "Screened 0.45. Fixed: one water-limitation direction is calibrated through soilWHC (phase 1 rec. 8; probe: raising it to 0.5 recovers less JJA uptake than soilWHC 88).",
    "water_use_efficiency": "Screened 0.43. Fixed at BETY median for the same reason (one water direction).",
    "leaf_carbon_per_area": "Screened 0.43 (LAI conversion). Fixed at the BETY median 30.7: BETY 2.5-97.5% is only +-12%.",
    "vapor_pressure_deficit_slope": "Screened 0.30. Fixed at BETY median.",
    "wood_turnover_rate": "Screened 0.29 (wood change only). Fixed at BETY median; AGB data (6 values over 6 years) barely separate it from wood allocation.",
    "wood_respiration_q10": "Screened 0.22. Fixed at BETY median; its effect is decorrelated from the 10 C rate by the re-referencing.",
    "soil_respiration_moisture_exponent": "Screened 0.19. Fixed at template 1.",
    "coarse_root_fraction": "Screened 0.19. Fixed at phase-1 0.16 (initial split only).",
    "leaf_turnover_rate": "Screened 0.18. Fixed at template 0.13 (phase 1: BETY 0.75 double counts leaf fall).",
    "leaf_off_day": "Screened 0.16. Fixed at template 285 (Harvard leaf fall DOY ~280-300, approximate).",
    "coarse_root_turnover_rate": "Screened 0.20. Fixed at template 0.056.",
    "fine_root_turnover_rate": "Screened 0.03. Fixed at BETY median.",
    "daily_mean_photosynthesis_fraction": "Degenerate with aMax/baseFolRespFrac/cFracLeaf (sipnet.c:614,617,633); fixed at BETY median, read by ComputePhotosynthesisRates.",
    "leaf_carbon_fraction": "Degenerate direction (as above); BETY median, read by ComputePhotosynthesisRates.",
}
f["justification"] = [why[n] + " | " + j if n in why else j for n, j in zip(f.flat_name, f.justification)]
f.to_csv(HERE / "fixed_parameters.csv", index=False)
print(f[["flat_name", "value", "morris_score"]].dropna().to_string(index=False))
