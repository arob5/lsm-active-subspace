| # | parameter | units | reaches SIPNET as | prior | median | source |
|---|---|---|---|---|---|---|
| 1 | `photosynthetic_capacity` | nmol g-1 s-1 | `aMax = P·cFracLeaf·(1−ρ)/aMaxFrac` | log-normal, 95% in [140, 450] | 251 | P2 |
| 2 | `respiration_share` | 1 | `baseFolRespFrac = ρ·aMaxFrac/(1−ρ)` | logit-normal, 95% in [0.04, 0.2] | 0.0926 | P2 |
| 3 | `optimum_photosynthesis_temperature` | degC | `psnTOpt` | N(22, 2.5²) | 22 | P2 |
| 4 | `photosynthesis_temperature_range` | degC | `psnTMin = psnTOpt − range` | log-normal, 95% in [17, 31] | 23 | BETY psnTMin 2.5-97.5% (-6.5, 6.6) below psnTOpt 24 |
| 5 | `half_saturation_light` | mol m-2 d-1 | `halfSatPar` | log-normal, 95% in [4.6, 26.3] | 11 | BETY |
| 6 | `vapor_pressure_deficit_slope` | kPa-1 | `dVpdSlope` | log-normal, 95% in [0.02, 0.12] | 0.049 | BETY median 0.050; R interval |
| 7 | `vapor_pressure_deficit_exponent` | 1 | `dVpdExp` | log-normal, 95% in [1, 3] | 1.73 | Niwot range [1, 3]; BETY 1.5, Braswell 2 |
| 8 | `light_extinction_coefficient` | 1 | `attenuation` (with `leafCSpWt` fixed) | log-normal, 95% in [0.38, 0.65] | 0.497 | BETY attenuation widened by BETY leafCSpWt spread (leafCSpWt fixed) |
| 9 | `water_use_efficiency` | mg g-1 kPa | `wueConst` | log-normal, 95% in [7.9, 17.5] | 11.8 | BETY |
| 10 | `water_removal_fraction` | d-1 | `waterRemoveFrac` | log-normal, 95% in [0.03, 0.25] | 0.0866 | T 0.088; R interval |
| 11 | `soil_water_holding_capacity` | cm | `soilWHC` | log-normal, 95% in [15, 150] | 47.4 | P2 |
| 12 | `interception_evaporation_fraction` | 1 | `immedEvapFrac` | logit-normal, 95% in [0.03, 0.3] | 0.103 | T 0.1; R interval |
| 13 | `fast_flow_fraction` | 1 | `fastFlowFrac` | logit-normal, 95% in [0.005, 0.3] | 0.0443 | T 0; R interval |
| 14 | `snow_melt_rate` | cm K-1 d-1 | `snowMelt` | log-normal, 95% in [0.05, 0.45] | 0.15 | T 0.15; R interval |
| 15 | `aerodynamic_resistance_constant` | 1 | `rdConst` | log-normal, 95% in [941, 1377] | 1.14e+03 | BETY |
| 16 | `soil_resistance_intercept` | 1 | `rSoilConst1` | N(8.2, 1²) | 8.2 | T 8.2; R sd |
| 17 | `soil_resistance_slope` | 1 | `rSoilConst2` | log-normal, 95% in [2, 9] | 4.24 | T 4.3; R interval |
| 18 | `leaf_on_growing_degree_days` | K d | `gddLeafOn` | log-normal, 95% in [500, 1100] | 742 | P2 |
| 19 | `leaf_off_day` | d | `leafOffDay` | N(285, 7²) | 285 | T 285; Harvard leaf fall DOY ~280-300 (approximate) |
| 20 | `leaf_on_growth` | g m-2 | `leafGrowth` | log-normal, 95% in [50, 180] | 94.9 | P2 |
| 21 | `leaf_off_fall_fraction` | 1 | `fracLeafFall` | logit-normal, 95% in [0.99, 0.9997] | 0.998 | BETY |
| 22 | `leaf_turnover_rate` | yr-1 | `leafTurnoverRate` | log-normal, 95% in [0.05, 0.35] | 0.132 | T 0.13; R interval (BETY 0.75 double-counts leaf fall) |
| 23 | `leaf_allocation` | 1 | `leafAllocation = ℓ` | logit-normal, 95% in [0.142, 0.217] | 0.176 | BETY |
| 24 | `wood_share_of_nonleaf_allocation` | 1 | `woodAllocation = (1−ℓ)·w` | logit-normal, 95% in [0.4, 0.7] | 0.555 | P2 centre 0.45/0.82 = 0.55; R interval |
| 25 | `fine_root_share_of_root_allocation` | 1 | `fineRootAllocation = (1−ℓ)(1−w)·f`; coarse roots get the rest | logit-normal, 95% in [0.08, 0.35] | 0.178 | P2 centre 0.065/0.37 = 0.18; R interval |
| 26 | `wood_turnover_rate` | yr-1 | `woodTurnoverRate` | log-normal, 95% in [0.00303, 0.00715] | 0.00465 | BETY |
| 27 | `fine_root_turnover_rate` | yr-1 | `fineRootTurnoverRate` | log-normal, 95% in [0.346, 7.19] | 1.58 | BETY |
| 28 | `coarse_root_turnover_rate` | yr-1 | `coarseRootTurnoverRate` | log-normal, 95% in [0.02, 0.15] | 0.0548 | T 0.056; R interval |
| 29 | `wood_respiration_rate_at_10c` | yr-1 | `baseVegResp = r10/vegRespQ10` | log-normal, 95% in [0.006, 0.04] | 0.0155 | P2 |
| 30 | `wood_respiration_q10` | 1 | `vegRespQ10` (wood and foliage) | log-normal, 95% in [1.449, 2.211] | 1.79 | BETY (also foliar respiration's Q10) |
| 31 | `fine_root_respiration_rate_at_10c` | yr-1 | `baseFineRootResp = r10/fineRootQ10` | log-normal, 95% in [0.15, 2.4] | 0.6 | BETY 0.186 yr-1 at 0 C x Q10 3.2 = 0.6; R factor-4 interval |
| 32 | `fine_root_respiration_q10` | 1 | `fineRootQ10` | log-normal, 95% in [1.511, 4.899] | 2.72 | BETY |
| 33 | `coarse_root_respiration_rate_at_10c` | yr-1 | `baseCoarseRootResp = r10/coarseRootQ10` | log-normal, 95% in [0.006, 0.04] | 0.0155 | P2 wood rate (coarse roots are wood), untied |
| 34 | `coarse_root_respiration_q10` | 1 | `coarseRootQ10` | log-normal, 95% in [1.521, 4.887] | 2.73 | BETY |
| 35 | `soil_respiration_flux_at_10c` | g m-2 yr-1 | `baseSoilResp = F10/(soilInit·soilRespQ10)` | log-normal, 95% in [200, 900] | 424 | P2 |
| 36 | `soil_respiration_q10` | 1 | `soilRespQ10` | log-normal, 95% in [1.3, 3.2] | 2.04 | P2 |
| 37 | `soil_respiration_moisture_exponent` | 1 | `soilRespMoistEffect` | log-normal, 95% in [0.3, 3] | 0.949 | T 1; R interval |
| 38 | `frozen_soil_threshold` | degC | `frozenSoilThreshold` | N(0, 1.5²) | 0 | T 0; Niwot range [-5, 5]; R sd |
| 39 | `frozen_soil_foliar_respiration_factor` | 1 | `frozenSoilFolREff` | U(0, 1) | 0.5 | T 0; no information |
| 40 | `frozen_soil_water_fraction` | 1 | `frozenSoilEff` | U(0, 1) | 0.5 | T 1, Niwot 0; no information |
| 41 | `initial_wood_carbon` | g m-2 | `plantWoodInit = W + C + F` | log-normal, 95% in [5000, 15000] | 8.66e+03 | IC median 8633; R interval |
| 42 | `initial_coarse_root_carbon` | g m-2 | `coarseRootFrac = C/(W + C + F)` | log-normal, 95% in [800, 3500] | 1.67e+03 | phase 1 1705; R interval |
| 43 | `initial_fine_root_carbon` | g m-2 | `fineRootFrac = F/(W + C + F)` | log-normal, 95% in [100, 800] | 283 | phase 1 320; R interval |
| 44 | `initial_soil_carbon` | g m-2 | `soilInit` | log-normal, 95% in [7000, 30000] | 1.45e+04 | IC median 15000; R interval |
| 45 | `initial_soil_wetness` | 1 | `soilWFracInit` | logit-normal, 95% in [0.3, 0.85] | 0.609 | IC median 0.61; R interval |
| 46 | `initial_snow_water_equivalent` | cm | `snowInit` | log-normal, 95% in [0.1, 8] | 0.894 | R |
