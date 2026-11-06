# Design 02 — The inverse problem: calibration parameters, prior, and problems

*Approved 2026-11-06 (draft 2). Draft 2 settled the prior's breadth, replaced the small problem
(§6.2), and said why allocation is not a Dirichlet (§2).*

Design 01 fixed how the pieces fit together, and used a four-parameter development problem to build
them. This design fixes the inverse problem itself: **which SIPNET parameters are calibrated, in
which coordinates, and with which prior**, and the two Harvard Forest problems built from them. The
data, the observation operator and the likelihood are design 01's and do not change (§5).

The rule, decided on 2026-10-15: **calibrate essentially every SIPNET parameter, and remove only
exact non-identifiability.** Near-degeneracies, weak parameters and ridges stay in, because finding
them is what the algorithms are for.

Every number below was produced by the prototype in
[`prototypes/02-inverse-problem/`](prototypes/02-inverse-problem/) at the pinned SIPNET
(`v2.2.0-alpha.1`), on the shipped driver, with `ModelFlags.standard()`. Results quoted as
"verified" come from those scripts.

**In one paragraph.** pySIPNET writes 58 parameters. Eight never affect NEE at this site and are
fixed. Three exact degeneracies remove three more, and the leafless start fixes the initial leaf
area. That leaves **46 calibration parameters**, given an independent prior assembled from the
phase-2 Harvard Forest priors, BETY, PEcAn's template and reasoned intervals, each with its source.
The prior predictive covers the observed features (95% of day and 94% of night features inside the
90% interval) with no failed runs. Averaged over the prior, the data inform twelve directions more
than the prior does, five of them strongly, and leave the rest weak: the structure the algorithms
must discover. A second,
smaller problem calibrates 10 of the 46, chosen so that a trustworthy reference posterior is
affordable and its informative directions mix its parameters: three directions hold 92% of its
information, the best three parameters only 65%.

---

## 1. From 58 SIPNET parameters to what the data can see

### 1.1 The active set

`t1_active_set.py` moves each of the 58 parameters both ways around a reference point and compares
the 400 features with the reference run. Eleven parameters reproduce the reference features bit for
bit:

| Why it is inert | Parameters |
|---|---|
| Its model option is off (`ModelFlags.standard()`: `gdd`, `snow`, `water_hresp` on) | `leafOnDay` (the GDD trigger decides leaf-on), `soilTempLeafOn`, `growthRespFrac`, `litterInit`, `litterBreakdownRate`, `fracLitterRespired`, `leafPoolDepth` |
| Its cap never binds at this site | `leafOnReallocFrac` (leaf-on growth is far below the wood pool it could draw on) |
| No leaves when the soil is frozen, **at the reference point only** | `frozenSoilThreshold`, `frozenSoilFolREff`, `frozenSoilEff` |

The last three are not inert across the prior: with an early leaf-on or a late leaf-off over cold
soil they act, and moving them far from their template values changed the features in 9, 6 and 6
of 30 prior draws (`t8_frozen_soil.py`). So they are **calibrated**. The first eight stayed inert in every one of 80 prior draws, each moved far from
its value (for example `leafOnReallocFrac` from 0.05 to 0.8, `leafOnDay` from day 100 to day 200).
They are **fixed**.

### 1.2 Exact degeneracies

Three directions leave the features exactly unchanged. `t2_degeneracies.py` checks each at
k = 0.97 and 1.03: the features match the reference bit for bit, while each parameter alone moves
them by 3–6%.

| Null direction (multiply or divide by k) | What the data see | Fixed |
|---|---|---|
| `aMax`·k, `cFracLeaf`·k | `aMax/cFracLeaf` | `cFracLeaf = 0.466` (BETY median) |
| `aMax`·k, `aMaxFrac`/k, `baseFolRespFrac`/k | `aMax·aMaxFrac/cFracLeaf`, `aMax·baseFolRespFrac/cFracLeaf` | `aMaxFrac = 0.861` (BETY median) |
| `leafCSpWt`·k, `attenuation`·k | `attenuation/leafCSpWt` | `leafCSpWt = 30.7 g C m⁻²` (BETY median) |

The third holds because the run starts leafless (`laiInit = 0`). The initial leaf area is fixed at 0
for that reason: 1 January at a deciduous forest has no leaves. With `laiInit` free, the third
direction would be `leafCSpWt`·k, `attenuation`·k, `laiInit`/k.

Are there others? `t3_jacobian_spectrum.py` takes the Jacobian of the features at the reference
point in log coordinates. In those coordinates an exact ratio degeneracy shows as a singular value
at round-off. Exactly two do, the two ratio directions above, at 1.7 × 10⁻¹⁴; the next is 1.7 × 10⁻³
(`snowMelt`, a real but tiny effect). The second direction is a sum, not a ratio, so finite
differences cannot show it exactly. Its residual, 0.23, equals the rounding floor predicted below
(0.23). Together with the source-level analysis of the parameter-structure note, this leaves the
three.

> **SIPNET writes NEE to three decimals** (`%8.3f`, `sipnet.c:461-466`): 0.001 g C m⁻² per 3-hour
> step, 0.0077 µmol m⁻² s⁻¹. For the likelihood this is negligible: a feature's rounding error has
> sd about 5 × 10⁻⁴ µmol m⁻² s⁻¹, against a noise sd of 1.2–2.8. For numerical derivatives it is
> not. A central difference of width 2h in a whitened coordinate carries rounding noise of about
> 0.002/h in the information-weighted units of §4, so steps below h ≈ 0.05 resolve nothing in the
> weak directions. Algorithms that difference SIPNET use h ≥ 0.1, or ensemble linearizations
> whose spread is the prior's. This is recorded as **SP-01** in the gap registry; the fix belongs in
> SIPNET's output format.

### 1.3 The count

| | Parameters |
|---|---|
| pySIPNET's parameters | 58 |
| inert at this site, fixed (§1.1) | −8 |
| fixed to remove an exact degeneracy (§1.2) | −3 |
| fixed by the leafless start | −1 |
| **calibrated** | **46** |

---

## 2. The calibration parameters

The 46 calibration parameters are SIPNET parameters, or **bijective** reparameterizations of them.
A reparameterization changes nothing the model can do. It is chosen so that a prior can be stated
independently and in units an ecologist recognizes:

1. **Photosynthetic capacity and respiration share** replace `aMax` and `baseFolRespFrac`, as in the
   development problem: P = aMax(aMaxFrac + baseFolRespFrac)/cFracLeaf and
   ρ = baseFolRespFrac/(aMaxFrac + baseFolRespFrac). Together with the fixed `aMaxFrac` and
   `cFracLeaf` they span exactly what the data see (§1.2).
2. **The optimum temperature and the temperature range** replace (`psnTMin`, `psnTOpt`), with
   `psnTMin = psnTOpt − range`. The range is positive, so `psnTMin < psnTOpt` always holds.
3. **Respiration rates at 10 °C** replace SIPNET's rates at 0 °C, for wood, fine roots, coarse roots
   and soil: `base = r10/Q10`. Measurements report rates near the site's temperatures, so a prior
   on the rate at 10 °C can be stated independently of the Q10; one on the rate at 0 °C cannot. For soil,
   the **flux** at 10 °C, F10 = k10 · soilInit, replaces the rate, as in phase 2, so that the soil
   pool and the soil respiration flux have independent priors.
4. **Allocation by stick-breaking**: the leaf fraction ℓ; the wood share w of the rest; the fine-root
   share f of what remains; coarse roots take the remainder. Any three values in (0, 1) give a valid
   allocation, so SIPNET's exit on an invalid allocation (`ensureAllocation`, `sipnet.c:1111-1123`)
   cannot happen. The three shares are three scalar parameters with independent priors, and the map
   is three lines: `leafAllocation = ℓ`, `woodAllocation = (1 − ℓ)w`,
   `fineRootAllocation = (1 − ℓ)(1 − w)f`. No bijector is implemented.

   *Why not a Dirichlet on ProbPipe's simplex.* ProbPipe has a `Dirichlet` law on its `simplex`
   support, with a centred-softmax map to unconstrained coordinates, so the four shares could be
   one vector parameter. A Dirichlet has a single total concentration α₀, which fixes every share's
   spread relative to its mean, Var xᵢ = mᵢ(1 − mᵢ)/(α₀ + 1). The sources do not agree with that:
   BETY pins the leaf share (sd 0.019, which needs α₀ ≈ 400), while the wood and root shares are
   uncertain (sds 0.064, 0.029, 0.058, which need α₀ ≈ 60–80). At α₀ = 400 the wood share's sd
   would be 0.025 instead of 0.064; at α₀ = 60 the leaf share's would be 0.049 instead of 0.019
   (verified, 200,000 draws). Independent shares are the generalized-Dirichlet construction, with
   logit-normal sticks in place of beta ones; ProbPipe does not offer it, and it needs nothing
   beyond scalar laws. Scalar parameters also keep the one-row-per-parameter prior table (§7) and
   the coordinate stopgap, which handles scalar parameters only (PG-05).
5. **Initial carbon pools** replace (`plantWoodInit`, `fineRootFrac`, `coarseRootFrac`):
   above-ground wood W, coarse roots C and fine roots F, each in g C m⁻², with
   `plantWoodInit = W + C + F` and the fractions C/(W + C + F) and F/(W + C + F).

None of these ties two parameters together. Phase 2's choices that did (one water direction through
`soilWHC` alone; one rate at 10 °C shared by wood and coarse roots) are **undone**: they were
reductions of approximate ridges, and finding approximate ridges is the algorithms' job.

Two parameters act as **step functions**: `gddLeafOn` and `leafOffDay` decide the leaf-on and
leaf-off days, so the features change only when a threshold crosses a time step. Along those two
axes the likelihood is piecewise constant, and a gradient is zero almost everywhere. They are
calibrated anyway: leaf-on timing was phase 2's largest single error, and `leafOffDay` is the
parameter that moves the features most at the reference point. Methods that read gradients must
treat them with care; ensemble linearizations, whose differences span many time steps, see them.
A finite difference sees them too, but what it sees depends on its width: their activity scores
grow by factors 1.3 and 2.1 when the width shrinks from 0.2 to 0.05 prior sds, while every smooth
parameter's changes by at most 4% (`t9_small_problem.py`).
`frozenSoilThreshold` is switch-like too, but its effect is rare (§1.1).

The full list, with priors, is in §3.

---

## 3. The prior

### 3.1 Choices

The prior is a product of independent one-dimensional laws, one per calibration parameter, in four
families: log-normal (positive quantities), logit-normal (fractions), normal (temperatures and the
soil-resistance intercept) and uniform (two frozen-soil fractions with no information). Log-normal
and logit-normal priors are stated by their central 95% interval, as `lsmas.priors` already does.
Independence is a statement of ignorance, not a modelling convenience: phase 2 found no evidence for
a correlation once the exact ridges were removed, and the reparameterizations of §2 put the
correlations that are known (base rate and Q10; soil rate and pool) into the coordinates.

Sources, in order of preference:

- **P2**: the phase-2 Harvard Forest prior (`design/inputs/harvard_forest/parameter_analysis/
  prior_proposal.csv`), itself checked against the site's data by prior-predictive iteration.
- **BETY**: the temperate-deciduous 2.5–97.5% after PEcAn's unit conversion
  (`base_parameter_candidates.csv`), with phase 1's corrections (PEcAn's root and stem respiration
  unit slip; the implausible BETY `psnTOpt` and `aMax`).
- **T**: PEcAn's SIPNET template value, with a reasoned interval around it.
- **IC**: the site's initial-condition ensemble median, with a reasoned interval.
- **R**: reasoned here, from the parameter's role.

### 3.2 The table

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

Fixed, with their values (`data/harvard_forest/sipnet_fixed.csv` already holds them):
`aMaxFrac` 0.861, `cFracLeaf` 0.466, `leafCSpWt` 30.7, `laiInit` 0, `leafOnDay` 144,
`soilTempLeafOn` 12, `growthRespFrac` 0.272, `litterInit` 0, `litterBreakdownRate` 0.4,
`fracLitterRespired` 0.5, `leafPoolDepth` 0.200, `leafOnReallocFrac` 0.2.

Three choices deserve a word:

- **`light_extinction_coefficient`** stands for `attenuation/leafCSpWt`, the combination the data
  see. With `leafCSpWt` fixed, its prior is BETY's `attenuation` interval widened by BETY's
  `leafCSpWt` spread (log-sds 0.116 and 0.063 combined).
- **The two frozen-soil fractions** have uniform priors: SIPNET's template and the Niwot file
  disagree (0 against 1), and nothing else informs them.
- **The phenology priors** keep phase 2's `gddLeafOn` interval, chosen from the driver's degree-days
  and the observed onset of uptake, and centre `leafOffDay` on the template's day 285 with sd 7 days
  (Harvard leaf fall is around days 280–300).

### 3.3 Prior predictive check

`t4_prior_predictive.py` draws 400 parameter sets from the prior, runs SIPNET at each, and adds the
likelihood's noise with scales drawn from their inverse-gamma prior.

**No run failed.** pySIPNET accepted every parameter set, and SIPNET finished every run.

**Coverage of the observed features**, the fraction inside the 90% interval of the replicated data,
with the mean standardized residual z = (y − median)/sd:

| class | all | DJF | MAM | JJA | SON | z (all) |
|---|---|---|---|---|---|---|
| day (214) | 0.95 | 0.94 | 1.00 | 0.89 | 0.98 | −0.40 |
| night (186) | 0.94 | 0.96 | 0.91 | 0.97 | 0.92 | +0.30 |

The data lie in the bulk. Two seasonal tendencies remain, as in phase 2: summer daytime uptake is
weak at the median (JJA day z = −1.03; 11% of features below the interval), and spring night
respiration is low (MAM night z = +0.73).

**Plausibility of the runs** (annual means over 2012–2020, g C m⁻² yr⁻¹; 5 / 50 / 95%):

| quantity | 5% | 50% | 95% |
|---|---|---|---|
| GPP | 546 | 1180 | 2040 |
| ecosystem respiration | 679 | 1100 | 1700 |
| NEE | −711 | −59 | +451 |
| Ra / GPP | 0.40 | 0.63 | 1.01 |
| annual peak LAI | 3.0 | 6.0 | 9.9 |

The tails are broad. 43% of runs make the site a net source of carbon over 2012–2020 (phase 2's
15-parameter prior: 34%), and 35% have GPP below 1000. The run at the prior **medians** is a
plausible sink (GPP 1421, NEE −377; `t5_median_shift.py`), so the source-heavy tail comes from
46 broad priors compounding, not from a misplaced centre. Varying only the phase-2 parameters gives
30% sources; varying only the others gives 16%; varying all gives 46% (`t6_group_spread.py`, 200
draws each). The largest single contributor is `half_saturation_light`, whose BETY interval spans a
factor of 5.7 (its rank correlation with NEE across the draws is 0.72 when only the non-phase-2
parameters vary).

The prior stays this broad (decided, §8). A broad prior costs nothing in the likelihood, which
dominates it, but it shapes the whitened coordinates in which subspaces are measured, and it sets
where EKI's ensemble starts; the walkthrough notebook shows the prior predictive so that this is
visible.

---

## 4. What the data inform

`t7_prior_importance.py` estimates the prior average of the tempered likelihood's information in
whitened coordinates,

$$
C = \mathbb E_{\pi}\!\left[J(\xi)^\top W J(\xi)\right],
\qquad W = \operatorname{diag}\!\left(\frac{\eta_i}{v_{k(i)}}\right),
$$

where J is the Jacobian of the 400 features in the standard-normal coordinates ξ of the independent
prior, by central differences (h = 0.2) at 80 prior draws. Its eigenvalues are the spectrum an
active-subspace method estimates, and its diagonal gives each parameter's activity score. These are
estimates for the design, not results the algorithms must reproduce: design 03 defines what they
estimate.

**The spectrum** (largest first):

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 20 | 30 | 40 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eigenvalue | 74.6 | 26.5 | 15.7 | 13.1 | 11.8 | 4.1 | 3.3 | 2.7 | 2.3 | 2.1 | 1.3 | 1.3 | 0.80 | 0.20 | 0.024 | 0.0013 |

An eigenvalue above 1 means that, in that direction, the tempered data carry more information
than the prior. Twelve directions do; five dominate; the last thirty carry almost nothing. One
eigenvalue is exactly zero: `frozen_soil_water_fraction` never moved the features under small steps
at any of the 80 draws. These numbers are Monte Carlo estimates: a 30-draw estimate with other
draws gave a leading eigenvalue of 78 and the same shape (`out/t7.txt`).

**Activity scores**, the diagonal of C, largest first:

| parameter | score | parameter | score |
|---|---|---|---|
| `photosynthetic_capacity` | 43.9 | `soil_respiration_flux_at_10c` | 3.4 |
| `soil_water_holding_capacity` | 19.4 | `water_removal_fraction` | 3.3 |
| `leaf_on_growing_degree_days` | 14.5 | `optimum_photosynthesis_temperature` | 3.1 |
| `half_saturation_light` | 14.1 | `light_extinction_coefficient` | 2.8 |
| `leaf_off_day` | 13.2 | `fine_root_respiration_rate_at_10c` | 1.7 |
| `leaf_on_growth` | 8.4 | `soil_respiration_q10` | 1.3 |
| `respiration_share` | 8.4 | `wood_respiration_rate_at_10c` | 1.2 |
| `water_use_efficiency` | 7.6 | `photosynthesis_temperature_range` | 1.0 |
| `vapor_pressure_deficit_slope` | 4.6 | (28 more) | < 1 |
| `vapor_pressure_deficit_exponent` | 4.0 | | |

The ten largest account for 84% of the trace. Three things here are worth an algorithm's attention:

- **The water directions are strong at the prior's scale.** `soilWHC`, `wueConst`,
  `waterRemoveFrac` and the two VPD parameters together score about 39. Phase 2 kept only `soilWHC`.
  They form the approximate ridge of the parameter-structure note (§3.5), which a good subspace
  estimator should find as a few combinations rather than five axes.
- **Two of the five largest are step functions** (`gddLeafOn`, `leafOffDay`). A finite difference
  of width 0.4 prior sds sees them; a gradient would not. Their two scores here are properties of
  that width, not of the model (§2).
- **Night respiration is spread thinly**: soil, root and wood respiration each score 1–3, and the
  base-rate–Q10 pairs are the classic weak ridge.

---

## 5. The likelihood (unchanged)

The likelihood is design 01's and the working-likelihood note's, and the code already implements it:

$$
y_i \mid \theta, s^2 \sim \mathcal N\!\left(G_i(\theta),\; s^2_{k(i)} v_{k(i)}\right),
\qquad s^2_k \sim \mathrm{IG}(2, 1.678),
\qquad v_{\mathrm{day}} = 8,\; v_{\mathrm{night}} = 1.5,
$$

tempered by η = 0.126 (day) and 0.137 (night), from c = 3 effective observations per class-year.

One caveat. c = 3 was calibrated at Harvard Forest by matching log-determinants of linearized
posterior precisions, using the Jacobian of phase 2's 15-parameter model. With 46 parameters the
matched learning rate could differ, because the prior covariance and the Jacobian enter the match.
The note recommends rerunning at c = 2 and c = 4 as a sensitivity check; with 46 parameters that
check matters more. This design keeps c = 3.

---

## 6. The problems

### 6.1 `harvard_forest`: the full problem

The 46 calibration parameters, the prior of §3, the likelihood of §5, and the noise scales either
fixed at 1 or inferred (`scales="fixed"` or `"inferred"`, as in the development problem).

```python
from lsmas.problems import load

problem = load("harvard_forest", scales="inferred")
print(problem.describe())
# harvard_forest: Harvard Forest NEE, 2012-2020, with 46 calibration parameters (noise scales inferred).
# Calibration parameters (46): photosynthetic_capacity, respiration_share, ...
# Observations: nee (400 values)
problem.coordinates.dimension     # 46
```

### 6.2 `harvard_forest_small`: a problem with a trustworthy reference

The full problem's reference posterior will be hard to trust (design 04). The small problem keeps
the same data, likelihood and prior, and fixes all but **ten** calibration parameters at their prior
medians. Four requirements chose the ten:

1. **Tractable.** Long MCMC must be affordable: at 0.3 s per run, 100,000 runs take about an hour
   with 8 chains running in parallel (ProbPipe's RWMH evaluates one point per chain at a time).
2. **Important.** Calibration should be worth doing: six of the eight largest activity scores (§4)
   are in, and the ten hold 71% of the full problem's trace of C.
3. **A reduction is more than a selection.** The parameters are chosen in groups that act
   together through the model (the correlations of C, §4), so that the informative directions mix
   parameters: photosynthetic capacity, the light half-saturation and the extinction coefficient
   (correlations −0.92 and +0.95 in C); the VPD slope and exponent (+0.96); the water-holding
   capacity and water-use efficiency (+0.84); the foliar respiration share and the soil flux
   (+0.55).
4. **Smooth.** The two leaf-timing parameters are fixed at their medians, because their
   derivatives are not defined and their finite-difference scores depend on the width (§2). The
   small problem is where gradient-based and gradient-free estimators are compared against a
   reference they can trust; a target that depends on the step width would confound that
   comparison. The full problem keeps them.

The ten: `photosynthetic_capacity`, `half_saturation_light`, `light_extinction_coefficient`,
`leaf_on_growth`, `vapor_pressure_deficit_slope`, `vapor_pressure_deficit_exponent`,
`soil_water_holding_capacity`, `water_use_efficiency`, `respiration_share`,
`soil_respiration_flux_at_10c`.

`t9_small_problem.py` checks the choice on the small problem itself: C computed with the other 36
parameters at their medians, at 40 prior draws, at widths 0.2 and 0.05 (the two agree to 2%). Its
eigenvalues are

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| eigenvalue | 63.1 | 27.6 | 14.0 | 3.2 | 2.3 | 1.8 | 1.3 | 0.32 | 0.17 | 0.09 |

and the information each reduction leaves out is

| dimension r | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| outside the top r eigenvectors | 45% | 20% | 8% | 5% |
| outside the best r parameters | 66% | 45% | 35% | 26% |

Three directions dominate, and each mixes parameters: carbon uptake (photosynthetic capacity −0.77,
half-saturation +0.40, leaf growth −0.29, water-holding capacity +0.23), water (water-holding
capacity +0.83, water-use efficiency +0.44) and respiration (respiration share +0.80, soil flux
+0.32). Four more exceed 1, so choosing the dimension is a real trade-off. The last three are the
near-ridges, and carry almost nothing: the VPD slope against the exponent (−0.72, +0.68), the
half-saturation against capacity and leaf growth, and the extinction coefficient against leaf
growth.

```python
small = load("harvard_forest_small")
small.coordinates.names
# ('photosynthetic_capacity', 'half_saturation_light', ..., 'soil_respiration_flux_at_10c')
```

### 6.3 The development problem retires

`harvard_forest_dev` was a stand-in for these two. Its tests move to `harvard_forest_small`, and the
module goes. (It also left `soilWFracInit` at pySIPNET's Niwot default; both new problems set it.)

### 6.4 The toy problem

The differentiable toy problem of design 01 serves the algorithms (it checks averaged-gradient
estimators against exact Jacobians), so it is specified with them, in design 03.

---

## 7. Code shape

One source of truth for the prior: **`data/harvard_forest/prior.csv`**, one row per calibration
parameter with its units, its family, the family's own numbers (`lower` and `upper` for the
interval families and the uniform, `mean` and `sd` for the normal) and its source (the table of
§3.2). `lsmas` reads it; nothing else states a prior value. The medians are not stored: they are
computed exactly from each law, which for a transformed law needs a stopgap (PG-16), since ProbPipe
estimates its quantiles by Monte Carlo.

| Module | Holds |
|---|---|
| `data/harvard_forest/prior.csv` | the 46 priors (new) |
| `data/harvard_forest/sipnet_fixed.csv` | the 12 fixed SIPNET values, each with its reason (updated: it now holds exactly the fixed parameters) |
| `lsmas.data` | `HarvardForestData.prior_table`, read from the CSV (new) |
| `lsmas.priors` | the families by interval (exists), plus `laws_from_table(table)`, one ProbPipe law per row, and `independent_prior(laws)`, their product grouped under `theta` (new) |
| `lsmas._stopgaps.quantiles` | `median(law)`, exact for a law transformed by a monotone bijector (PG-16, new) |
| `lsmas.parameters` | the maps of §2, one small function each, all exact: `photosynthesis_to_sipnet`, `soil_respiration_to_sipnet` (exist), `photosynthesis_temperatures_to_sipnet`, `respiration_rates_to_sipnet`, `allocation_to_sipnet`, `initial_pools_to_sipnet` (new), and `calibration_to_sipnet(theta, fixed)`, which composes them into the 58 SIPNET values and refuses a set that misses or repeats one |
| `lsmas.problems.harvard_forest` | `harvard_forest(scales=..., data=...)`, `harvard_forest_small(...)` and `SMALL_PROBLEM_PARAMETERS` (new) |
| `lsmas.problems.harvard_forest_dev` | deleted |

Each map in `lsmas.parameters` states its formula in its docstring and has a unit test against a
hand-computed value. Tests check that the prior table names exactly the parameters
`calibration_to_sipnet` reads, that each of them reaches SIPNET, and that every one of the 58 SIPNET
parameters is set exactly once. Smoke tests run the full problem's prior predictive at 16 draws and
check that no run fails, and check that the small problem's forward map is the full one's with the
other 36 parameters at their medians.

ProbPipe handles a 46-parameter prior without trouble: merging the 46 laws and renaming them under
`theta` takes 0.24 s, 100 draws 0.6 s, a `log_prob` 0.2 s, and `Coordinates.from_prior` (50,000
draws) 1 s (verified). `Uniform` works with the coordinate stopgap: its support's bijector is the
logistic function.

---

## 8. Decisions (approved 2026-11-06)

1. **The parameter set.** 46 calibration parameters as in §1–2: everything that is not exactly inert
   or exactly degenerate, including the step functions and the three frozen-soil parameters.
2. **The reparameterizations of §2**, all bijective, none tying parameters; allocation as three
   scalar shares, not a Dirichlet.
3. **The prior's breadth**: kept as drafted (43% of prior runs are net sources; phase 2: 34%),
   with the prior predictive shown in the walkthrough notebook.
4. **The small problem** (§6.2): the ten parameters chosen by the four requirements there.
5. **c = 3 with 46 parameters** (§5): kept for now; the c = 2 and c = 4 check runs in the
   experiments.
6. **`harvard_forest_dev` retires** (§6.3), and **the toy problem moves** to design 03 (§6.4).

---

## Appendix: what the prototype verified

| Claim | Script | Result |
|---|---|---|
| 11 parameters inert at the reference point | `t1_active_set.py` | bit-identical features under both perturbations |
| 8 stay inert across the prior; 3 frozen-soil ones do not | `t7_prior_importance.py`, `t8_frozen_soil.py` | 0 of 80 draws moved for the 8; 9, 6, 6 of 30 for the frozen-soil ones |
| Three exact degeneracies | `t2_degeneracies.py` | relative change 0 at k = 0.97 and 1.03; 3–6% for each parameter alone |
| No further exact ratio degeneracy | `t3_jacobian_spectrum.py` | two singular values at 1.7 × 10⁻¹⁴, the next 1.7 × 10⁻³ |
| SIPNET's output rounding | `t3_jacobian_spectrum.py` (session check) | NEE quantum 0.001 g C m⁻² per step; finite-difference floor 0.23 at h = 0.01 |
| Prior predictive | `t4_prior_predictive.py` | 400 runs, 0 failures; coverage 0.95 day, 0.94 night |
| Median run is a plausible sink | `t5_median_shift.py` | GPP 1421, NEE −377 g C m⁻² yr⁻¹ |
| Spread by group | `t6_group_spread.py` | sources: 30% (phase-2 group), 16% (others), 46% (all) |
| Prior-averaged information | `t7_prior_importance.py` | 80 draws: 12 eigenvalues above 1, leading 74.6; the eight fixed-as-inert parameters inert at every draw |
| Frozen-soil parameters live in parts of the prior | `t8_frozen_soil.py` | moved the features at 9, 6, 6 of 30 draws |
| A single Dirichlet cannot hold the allocation prior | (session check) | each share's sd needs α₀ ≈ 400 (leaf) or 59–77 (wood, roots) |
| The small problem's directions mix its parameters; the leaf-timing scores depend on the width | `t9_small_problem.py` | 3 directions leave 8% outside, the best 3 parameters 35%; score ratios (h 0.05 / 0.2) 1.31 and 2.05 for the leaf timing, ≤ 1.04 otherwise |
| ProbPipe with 46 parameters | (session check) | build 0.24 s, coordinates 1 s, `Uniform` supported |
