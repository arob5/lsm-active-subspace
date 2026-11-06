# Design 03 — The algorithms: subspaces, reduced posteriors, and the first algorithm

*Approved 2026-11-06 (draft 2). Draft 2 states how the toy differs from SIPNET (§1) and records two decisions:
the toy's docstring explains its model in full, and the sampler's sequential chains wait (§6).*

Designs 01 and 02 fixed the pieces and the inverse problem. This design fixes the algorithmic side:
**what an active subspace estimates here, how the shipped tools estimate it, how a reduced posterior
is defined and sampled, and the one complete algorithm the repository ships**. It also specifies the
differentiable toy problem, because exact Jacobians are what the claims about estimators are checked
against, and EKI with inferred noise scales, which every algorithm needs for the full likelihood.

Every number below comes from the prototype in
[`prototypes/03-algorithms/`](prototypes/03-algorithms/), at the pinned packages. Most checks run on
the toy, where NUTS gives a reference posterior and JAX gives exact Jacobians; the last runs the
proposed algorithm on `harvard_forest_small`.

**In one paragraph.** The toy is SIPNET's canopy equations in JAX: 14 of the shipped calibration
parameters with their priors, the real driver, operator, data and likelihood, a run in 0.9 ms, and a
NUTS reference in 11 minutes. The quantity a subspace estimates is the posterior average of the
tempered Gauss–Newton information in whitened coordinates. EKI estimates it at no extra cost by
accumulating its statistical linearizations over the tempering ladder: on the toy those directions
hold 98–99% of the best possible information at every rank, though the eigenvalues come out about
40% low. The reduced posterior keeps the inactive variables at their prior; sampling its active part
with ProbPipe's own random-walk Metropolis, at the rank the eigenvalues suggest, beats both EKI alone
and full-dimensional random-walk Metropolis at the same budget. The shipped algorithm,
`active_subspace_mcmc`, is that pipeline: explore with EKI, estimate the subspace, sample the reduced
posterior, lift it back. On `harvard_forest_small` it runs end to end, with interpretable directions.

---

## 1. The toy problem

**What it is for.** Two things the SIPNET problems cannot give: exact Jacobians, to check what the
gradient-free estimators estimate, and a cheap trustworthy reference posterior (NUTS through plain
`condition`), to score reduced posteriors.

**What it is.** A simplified SIPNET built from SIPNET's own canopy flux equations
(`sipnet.c:600-640`, the 6-layer light model of `calcLightEff`, the Q10 respiration terms), in JAX.
It is not SIPNET's canopy component: it drops the pools and the water balance (below). Per 3-hour
step, with the shipped driver:

$$
\begin{aligned}
C_L &= g \;\sigma\!\left(\tfrac{\mathrm{GDD} - \mathrm{GDD}_{\mathrm{on}}}{30}\right)
       \sigma\!\left(\tfrac{d_{\mathrm{off}} - \mathrm{doy}}{5}\right)
       &&\text{leaf carbon, g C m}^{-2}\\
\mathrm{GPP} &= P \cdot 1.2\!\times\!10^{-8} \cdot 86400 \cdot C_L \;
       D_T(T_a)\, D_{\mathrm{VPD}}(\mathrm{VPD})\, D_L(C_L, \mathrm{PAR})
       &&\text{SIPNET's } D_T, D_{\mathrm{VPD}}, D_L\\
R &= \rho\,P \cdot 1.2\!\times\!10^{-8} \cdot 86400 \cdot C_L \cdot 1.8^{(T_a - T_{\mathrm{opt}})/10}
     + k_{10} W\, 1.8^{(T_a - 10)/10}/365 + F_{10}\, Q_{10}^{(T_s - 10)/10}/365 \\
\mathrm{NEE} &= R - \mathrm{GPP}, \text{ as a rate in } \mu\text{mol m}^{-2}\text{ s}^{-1}
\end{aligned}
$$

$P$ and $\rho$ enter exactly as in SIPNET after design 02's photosynthesis map (the leaf carbon
fraction cancels). The 14 calibration parameters are rows of the shipped prior table, with their
priors: `photosynthetic_capacity`, `respiration_share`, `optimum_photosynthesis_temperature`,
`photosynthesis_temperature_range`, `half_saturation_light`, `vapor_pressure_deficit_slope`,
`vapor_pressure_deficit_exponent`, `light_extinction_coefficient`, `leaf_on_growing_degree_days`,
`leaf_off_day`, `leaf_on_growth`, `wood_respiration_rate_at_10c`, `soil_respiration_flux_at_10c`,
`soil_respiration_q10`. The observation operator, the 400 observed features, the likelihood and the
learning rate are the Harvard Forest problem's.

**Verified** (`toy.py`, `t1_toy_nuts.py`):

| | |
|---|---|
| at the prior medians | GPP 1264, NEE −280 g C m⁻² yr⁻¹; correlation with the observed features 0.93 |
| one run / full Jacobian (400 × 14) | 0.9 ms / 10 ms |
| NUTS through `problem.condition(method="blackjax_nuts")` | 4 × 1000 draws, 677 s (885 s with inferred scales); R-hat ≤ 1.001 for every parameter |
| what the data inform | 7 eigenvalues of the posterior information above 1 (§2); posterior sd / prior sd from 0.43 to 0.99 per parameter |

So the toy is not easy: the data inform about seven directions, none of them a single parameter.

**How it differs from SIPNET.** Taken from SIPNET's source: the temperature, VPD and light effects,
gross photosynthesis as capacity times those effects times the leaf-carbon conversion (with $P$
and $\rho$ as after design 02's map), and the foliar, wood and soil respiration terms with their
Q10 forms. Simplified or dropped: leaf carbon is a smooth leaf-on/leaf-off curve of fixed amplitude
instead of SIPNET's leaf pool (leaf-on growth, allocation, turnover, leaf fall); there is no water
stress on photosynthesis and no moisture effect on soil respiration; the wood pool is fixed at
8660 g C m⁻² and the soil flux is referenced to a constant pool; the foliar and wood Q10 are fixed
at 1.8; roots, frozen soil, snow and output rounding are omitted. Against SIPNET at the prior
medians (all 46 parameters for SIPNET, the 14 for the toy; session check):

| | toy against SIPNET |
|---|---|
| GPP per 3-hour step | correlation 0.974; annual GPP 1264 against 1621 g C m⁻² |
| the 400 NEE features | correlation 0.979 (day 0.970, night 0.978) |
| mean absolute feature difference | day 1.68 against a mean SIPNET magnitude of 6.09; night 0.46 against 2.46 µmol m⁻² s⁻¹ |

The patterns agree closely; the toy's GPP is about 22% lower, mostly because its leaf carbon does
not grow after leaf-on. The toy's module docstring states the model in full, equations and
departures from SIPNET, so that a user reads what it is without opening SIPNET's source.

**The data.** The toy uses the real observations, so that everything except the forward map is the
real problem's. A synthetic-twin option (the toy at a fixed prior draw, plus the likelihood's noise)
is cheap to add if a known truth is wanted (decision 1).

---

## 2. What a subspace estimates

The notation is that of [`docs/notation.md`](../docs/notation.md), which every docstring follows.
Algorithms work in whitened coordinates $\xi$ (design 01 §4.1), where the prior is close to
$\mathcal N(0, I)$ (exactly, for the log-normal, logit-normal and normal priors). With $DG(\xi)$ the
Jacobian of the 400 features and $\Gamma = \operatorname{diag}(s^2_{k(i)} v_i / \eta_i)$ the
tempered noise covariance, the tempered likelihood's Gauss–Newton information at $\xi$ is
$DG(\xi)^\top \Gamma^{-1} DG(\xi)$. An active subspace is the span of the top eigenvectors of its
average over some law $\mu$,

$$
H(\mu) = \mathbb E_{\mu}\!\left[DG(\xi)^\top \Gamma^{-1} DG(\xi)\right].
$$

Because the prior's information in these coordinates is the identity, an eigenvalue above 1 means
the data inform that direction more than the prior does. **The reference is the posterior,**
$H_{\mathrm{post}} = H(\pi(\cdot \mid y))$: it is the Gauss–Newton likelihood-informed subspace of
Cui et al. (2014). Zahm et al. (2022) bound the KL error of the reduced posterior of §4 by half the
discarded eigenvalues of the posterior average of $\nabla \log L\, \nabla \log L^\top$; $H$ is its
usual Gauss–Newton stand-in, which needs no data residuals. The prior average $H(\pi)$, the
classical global active subspace, is a second reference.

**What one statistical linearization gives.** EnsKit's statistical linearization of an ensemble,
$\widehat{DG} = C_{G\xi}\, C_{\xi\xi}^{-1}$, equals the average Jacobian $\mathbb E_\mu[DG]$ when
the ensemble is Gaussian (Stein's lemma; EnsKit's documentation states it). So
$\widehat{DG}^\top \Gamma^{-1} \widehat{DG}$ estimates

$$
\bar H(\mu) = \mathbb E_\mu[DG]^\top \Gamma^{-1}\, \mathbb E_\mu[DG]
= H(\mu) - \mathbb E_\mu\!\left[(DG - \mathbb E_\mu DG)^\top \Gamma^{-1} (DG - \mathbb E_\mu DG)\right]
\;\preceq\; H(\mu),
$$

the information of the *average* Jacobian, which misses what varies across $\mu$. On the toy's prior
the two are verified to agree as Stein says (4000 draws: leading eigenvalue 42.8 from the
linearization, 43.2 from the average of exact Jacobians), and the gap is real: $H(\pi)$ has
eigenvalues 53.0, 20.0, 11.2, 4.8, $\bar H(\pi)$ 43.2, 16.4, 4.1, 2.4 (`t2_estimators.py`).

---

## 3. Estimators

### 3.1 The shipped estimator: linearizations accumulated over the EKI ladder

Tempered EKI evaluates the forward map on an ensemble at levels $0 = \beta_0 < \dots < \beta_T < 1$
and takes increments $\Delta\beta_t$. A callback linearizes each step's ensemble and accumulates

$$
\hat H_{\mathrm{ladder}} = \sum_{t=0}^{T} \Delta\beta_t\, \widehat{DG}_t^\top \Gamma_t^{-1} \widehat{DG}_t,
$$

a Riemann sum over the ladder of $\bar H(\mu_\beta)$, with $\mu_\beta$ the tempered posterior EKI's
ensemble approximates at level $\beta$. It costs no runs beyond EKI's, and it weights each level by
how much of the likelihood that step conditioned on.

**How good it is** (`t3_compare.py`). Score an estimate by the share of the posterior's best rank-$r$
information its top $r$ directions capture, $\operatorname{tr}(W^\top H_{\mathrm{post}} W) /
\sum_{i \le r} \lambda_i(H_{\mathrm{post}})$. Means over EKI seeds:

| estimate | r = 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| **ladder, J = 100** (5 seeds) | 0.981 | 0.984 | 0.987 | 0.986 | 0.980 | 0.987 |
| ladder, J = 50 (3 seeds) | 0.984 | 0.981 | 0.986 | 0.985 | 0.980 | 0.984 |
| ladder, J = 200 (3 seeds) | 0.985 | 0.988 | 0.991 | 0.990 | 0.982 | 0.989 |
| last level only, J = 100 | 0.898 | 0.960 | 0.984 | 0.984 | 0.979 | 0.986 |
| prior only, J = 100 | 0.925 | 0.948 | 0.950 | 0.964 | 0.970 | 0.979 |
| covariance-informed, last ensemble, J = 100 | 0.965 | 0.968 | 0.977 | 0.978 | 0.969 | 0.975 |
| exact $H$ of the prior (4000 draws) | 0.980 | 0.966 | 0.972 | 0.979 | 0.974 | 0.989 |

(The covariance-informed estimate is $\Sigma^{-1} - I$ for the ensemble's whitened covariance
$\Sigma$, exact for a linear model; Polette et al. 2026.) The ladder is the most stable of the
gradient-free estimates, at every rank and ensemble size, and as good as the exact prior average.

**Its eigenvalues are biased low.** Averaged over seeds at J = 100:

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | above 1 |
|---|---|---|---|---|---|---|---|---|---|
| $H_{\mathrm{post}}$ (exact) | 76.7 | 12.9 | 12.7 | 3.6 | 2.7 | 2.1 | 1.2 | 0.6 | 7 |
| exact $H$ along the ladder | 73.0 | 13.6 | 12.5 | 4.1 | 2.8 | 1.9 | 1.6 | 0.9 | 7 |
| $\bar H$ along the ladder | 70.3 | 11.8 | 9.5 | 3.4 | 1.7 | 0.8 | 0.2 | 0.2 | 5 |
| **$\hat H_{\mathrm{ladder}}$** | 43.3 | 9.0 | 8.5 | 3.1 | 1.6 | 0.7 | 0.3 | 0.2 | 5 |

Two effects: the average-Jacobian gap of §2, and, past the prior, ensembles that are not Gaussian,
so the least-squares slope is not the average Jacobian. The directions survive; the magnitudes do
not. That matters for choosing the rank (§3.3).

### 3.2 Other estimators, for comparison

All return the same `Subspace` (design 01 §4.7):

- **exact** (toy only): $H(\mu)$ from JAX Jacobians at given draws, prior or posterior;
- **finite differences** (SIPNET): $H(\mu)$ from central differences at given draws, $2d + 1$ runs per
  draw, width at least 0.1 prior sds because SIPNET rounds its output (design 02, SP-01);
- **last level**, **prior only**, **uniform weights**: variants of §3.1's accumulator;
- **covariance-informed**: from any ensemble or posterior sample.

### 3.3 Choosing the rank

Thresholds on the trace fail here: one direction dominates, so 95% of the trace is reached at r = 4
or 5 while seven directions are informed. The rule follows the reduced posterior's error bound
instead: **the smallest r whose discarded eigenvalues sum to at most ½**. On the ladder estimate it
gives r = 7 in all five seeds (on $H_{\mathrm{post}}$ itself it would give 10; with the bound 1, 6 and
9). The tolerance is a parameter, and the rank can always be given directly.

---

## 4. The reduced posterior

Write $\xi = W a + W_\perp b$, with $W$ the $d \times r$ subspace and $W_\perp$ an orthonormal
complement, so that $a$ and $b$ are independent $\mathcal N(0, I)$ under the prior. A reduced posterior
keeps $b$ at its prior and approximates the law of $a$. Three candidates were compared on the toy
against NUTS (`t4_reduced.py`, `t6_probpipe_reduced.py`):

- **optimal**: $a$ from the exact posterior's $a$-marginal, $\pi(a)\,\mathbb E_b[L(Wa + W_\perp b)]$.
  For a given $W$ this is the closest reduced posterior in KL (Zahm et al. 2022); here it is read off
  the NUTS draws, so it is a benchmark, not a method.
- **pseudo-marginal**: the same target, sampled by random-walk Metropolis whose likelihood is
  $L(Wa + W_\perp b)$ at one fresh prior draw of $b$, an unbiased estimate (one run per step).
- **plug-in**: $\pi(a)\, L(Wa)$, with $b$ at its prior mean inside the likelihood (Constantine, Kent &
  Bui-Thanh 2016). Its forward map $a \mapsto G(\theta(Wa))$ is deterministic.

Score: the mean over the 14 whitened coordinates of |mean error| / posterior sd, and of
|log sd ratio|, against NUTS. Baselines: the prior scores 0.81 / 0.29; EKI's final ensemble
(J = 100) 0.26 / 0.12 (J = 50: 0.30 / 0.18; J = 200: 0.19 / 0.10).

| r | optimal | plug-in | pseudo-marginal acceptance | variance of its log-likelihood |
|---|---|---|---|---|
| 1 | 0.74 / 0.24 | 0.73 / 0.24 | 0.00 | 995 |
| 2 | 0.70 / 0.21 | 0.70 / 0.21 | 0.01 | 292 |
| 4 | 0.73 / 0.12 | 0.73 / 0.13 | 0.01 | 138 |
| 6 | 0.14 / 0.07 | 0.16 / 0.07 | 0.13 | 11 |
| 8 | 0.10 / 0.04 | 0.11 / 0.04 | 0.15 | 2.6 |

(W from the ladder estimate, J = 100; the optimal column shows the limit of keeping $b$ at its prior.)

Three conclusions:

1. **The rank must cover what the data inform.** Below r = 6 every reduced posterior, even the
   optimal one, is worse than EKI's own ensemble: the discarded directions are informed, and the prior
   is a poor stand-in for them. At r = 6 to 8 the reduced posteriors beat EKI by about a factor 2 in
   both scores. The rule of §3.3 lands there.
2. **The plug-in is nearly optimal** at every rank, and it is the only one of the three that a
   standard sampler can run.
3. **Pseudo-marginal sampling is impractical here.** The estimate's log-likelihood variance must be
   of order 1 for the chain to move (Doucet et al. 2015); it is 10–1000 until r = 8, and the chains
   barely accept.

**The plug-in reduced problem is plain ProbPipe.** It is an `InverseProblem` whose parameters are the
active variables $a_1, \dots, a_r$ with independent $\mathcal N(0,1)$ priors, whose forward
`Function` is $a \mapsto G(\theta(Wa))$, and whose likelihood, data, learning rate and noise scales
are the full problem's. Conditioning it with `blackjax_rwmh` needs nothing new (verified on the toy
and on SIPNET; §6). The posterior over $\theta$ is its draws lifted: $\xi = Wa + W_\perp b$ with $b$
from the prior. Against the same budget of evaluations:

| on the toy, against NUTS | mean error | sd error | time |
|---|---|---|---|
| reduced plug-in, r = 7, `blackjax_rwmh` defaults | 0.143 | 0.056 | 14 s |
| reduced plug-in, r = 6 / 8 / 10 | 0.168 / 0.138 / 0.147 | 0.086 / 0.048 / 0.090 | |
| reduced plug-in, r = 7, noise scales inferred | 0.162 | 0.081 | 12 s |
| full 14-dimensional `blackjax_rwmh`, same defaults | 0.352 | 0.365 | 14 s |
| EKI alone, J = 100 | 0.26 | 0.12 | 3 s |

With inferred scales the reduced sampler's noise-scale intervals match NUTS's (day 5/50/95%:
0.59, 0.94, 1.66 against NUTS's 0.59, 0.93, 1.55; night 0.92, 1.41, 2.41 against 0.89, 1.42, 2.41).

---

## 5. EKI with inferred noise scales

The full likelihood infers the day and night noise scales with $\mathrm{IG}(2, 1.678)$ priors, and
`enskit_eki` today refuses that. The extension is the EKI-Gibbs algorithm of the Harvard Forest work
(Botha et al. 2023, with the noise step an exact draw), with the learning rate folded in. The
tempered target at level $\beta$ is

$$
\pi_\beta(z, s^2) \propto \pi(z) \prod_k \mathrm{IG}(s^2_k; \alpha_k, b_k)
\prod_i \mathcal N\!\left(y_i;\, G_i(z),\, s^2_{k(i)} v_i\right)^{\beta\, \eta_i}.
$$

Each step, at an ensemble $(z_j, s^2_j)$ whose predictions $g_j$ are known, with
$n_k = \sum_{i \in k} \eta_i$ and $q_{jk} = \sum_{i \in k} \eta_i (y_i - g_{ji})^2 / v_i$:

1. **Increment**: the largest $\Delta\beta \le 1 - \beta$ keeping the effective sample size of
   the weights $\log w_j = \sum_k [A_k(\beta) \log B_{jk}(\beta) - A_k(\beta')
   \log B_{jk}(\beta')]$ at least $J/2$, with $A_k(\beta) = \alpha_k + \beta n_k / 2$ and
   $B_{jk}(\beta) = b_k + \beta q_{jk} / 2$: the ratio of the targets with the scales integrated
   out.
2. **Parameter step**: EnsKit's `eki.assimilate` with noise $R(\bar s)/\Delta\beta$,
   $R(s)_{ii} = s^2_{k(i)} v_i / \eta_i$, for the common scales
   $\bar s^2_k = (\frac1J \sum_j s^{-2}_{jk})^{-1}$ (the "common gain").
3. **Evaluate** the new particles: EnsKit's `eki.evaluate`, the step's one batch of runs.
4. **Scale step**, exact: $s^2_{jk} \sim \mathrm{IG}(A_k(\beta'), B_{jk}(\beta'))$.

It is built from EnsKit's step-level API (`evaluate`, `assimilate`); EnsKit needs no change. The
final ensemble is evaluated once more, for its scale draws, so a run costs one batch more than
fixed-scale EKI.

**Verified on the toy** (`t5_eki_gibbs.py`, J = 100, three seeds, 2.6 s each): the noise scales'
5/50/95% intervals match NUTS (day 0.64–0.68 / 0.95–1.05 / 1.40–1.76 against 0.59 / 0.93 / 1.55;
night 0.87–0.96 / 1.39–1.49 / 2.27–2.43 against 0.89 / 1.42 / 2.41), and the parameters are as
accurate as fixed-scale EKI's (0.20–0.37 / 0.09–0.11).

**Step records gain the scales.** `StepRecord` adds `noise_scales`, the $(J, K)$ draws the step used,
and the accumulator of §3.1 uses $\bar s^2$ in $\Gamma_t$.

---

## 6. The shipped algorithm: `active_subspace_mcmc`

```python
from lsmas.algorithms import active_subspace_mcmc
from lsmas.problems import load

problem = load("harvard_forest_small", scales="inferred")
result = active_subspace_mcmc(problem, n_particles=100, seed=1)
result.subspace.eigenvalues        # the ladder estimate's spectrum
result.subspace.rank               # 6 here, by the rule of §3.3
result.posterior                   # draws of theta (and the noise scales), lifted to all parameters
result.cost                        # SIPNET runs by stage
```

1. **Explore.** Tempered EKI (EKI-Gibbs when the scales are inferred) with J particles, and the
   ladder accumulator of §3.1 as a callback.
2. **Estimate.** $W$: the top eigenvectors of $\hat H_{\mathrm{ladder}}$, at the rank of §3.3 unless
   one is given.
3. **Reduce.** The plug-in reduced problem of §4.
4. **Sample.** `blackjax_rwmh` on the reduced problem, started at the projection of EKI's final
   ensemble mean, with its proposal covariance from the projected final ensemble. Both are
   `blackjax_rwmh` options (verified); that the warm start shortens warmup is to be checked in
   implementation. The prototype ran the defaults.
5. **Lift.** $\xi = Wa + W_\perp b$, $b$ from the prior; back to $\theta$.

It returns an `AlgorithmResult` (design 01 §4.8): posterior, subspace, cost in runs, and the EKI
history.

**Verified on `harvard_forest_small`** (`t7_sipnet_small.py`, fixed scales, J = 100):

| | |
|---|---|
| EKI | 600 SIPNET runs, 72 s (8 threads) |
| ladder spectrum | 39.1, 8.45, 6.03, 1.71, 0.92, 0.29, 0.15, 0.12, 0.07, 0.04 → rank 6 |
| leading directions | uptake (capacity −0.78, half-saturation +0.51, extinction +0.24); water (water-holding capacity −0.84, water-use efficiency −0.50); respiration (respiration share +0.64, soil flux +0.60) — design 02's three groups |
| agreement with design 02's prior finite-difference matrix | the ladder's top 1, 2, 3, 6 directions hold 0.91, 0.95, 0.96, 0.99 of its best information |
| reduced `blackjax_rwmh` | 0.41 s per step: SIPNET's forward is not traceable, so chains run one after another, one point at a time |

At ProbPipe's defaults (4 chains of 500 warmup and 1000 draws) the sampling stage is about 6000 runs,
40 minutes; EKI's 600 runs are the smaller cost. Within a chain the steps are sequential, as
Metropolis requires, but ProbPipe's path for untraceable targets also runs the chains one after
another; run in lockstep, with each step's proposals evaluated as one batch through the forward
map's executor, the chains would share the time (PG-07, extended below). This waits: the SIPNET
reference posteriors will be computed once and stored (design 04), so the sampler's speed matters
only for the algorithms' own runs, and 40 minutes is acceptable there for now.

---

## 7. Code shape

As built:

| Module | Holds |
|---|---|
| `docs/notation.md` | the notation every docstring follows (new) |
| `lsmas.problems.nee_likelihood` | `nee_problem(name, description, forward, laws, scales=..., data=...)`: the likelihood, noise-scale priors and learning rate every problem shares (new; the Harvard Forest problems use it) |
| `lsmas.problems.toy` | `toy(scales=..., data=...)` and `ToyCanopyModel`, the model of §1, whose module docstring states its equations and its departures from SIPNET; registered as `"toy"` (new) |
| `lsmas.inference.noise_scales` | `NoiseScaleConditionals`: the scales' conditional laws, the increment weights with the scales integrated out, the common scales (new) |
| `lsmas.inference.enskit_eki` | one loop on EnsKit's step API for fixed and inferred scales (§5); `StepRecord` gains `noise_scales` and an `n_failed` read from EnsKit (updated) |
| `lsmas.tools.subspace` | `Subspace` (eigenvalues, eigenvectors, rank, `basis`, `complement`, active and inactive variables, loadings, activity scores, `information`) and `suggest_rank` (new) |
| `lsmas.tools.estimators` | `LadderLinearization` (weighting `"increment"`, `"uniform"` or `"last"`), `exact_subspace`, `finite_difference_subspace`, `covariance_subspace`, `statistical_linearization` (new) |
| `lsmas.tools.reduced` | `reduced_problem(problem, subspace)` and `ReducedProblem.lift(posterior, key=...)` (new) |
| `lsmas.algorithms` | `active_subspace_mcmc` and `AlgorithmResult` (new) |
| `lsmas.priors` | `factors_over(prior, outputs)`: the factors of a product prior over given outputs, such as the noise scales' priors (new) |
| `lsmas.problems.inverse_problem` | `InverseProblem.noise_variance(scale_values)`, the diagonal of $\Gamma$, which says which scale values are missing (updated) |
| `lsmas._stopgaps.coordinates` | maps between $\theta$ and $\xi$ in one call, and `parameter_record(xi)`, which JAX can trace (updated) |
| `lsmas._stopgaps.functions` | `declares_traceable(function)`, PG-18 (new) |

The sampler starts at the projection of EKI's ensemble mean and keeps ProbPipe's adaptive warmup.
On the toy that was both the most accurate and the steadiest of the variants tried (mean error
0.131–0.137, acceptance 0.29–0.31, against 0.141–0.144 and 0.14–0.27 from a prior draw); handing
it EKI's covariance as the proposal with a 100-step warmup was as accurate for 27% fewer runs, a
variant to try through `sampler_options`.

## 8. ProbPipe gaps

- **PG-07, extended.** With a forward map that is not traceable, `blackjax_rwmh` runs its chains one
  after another (the eager path loops over chains), as well as one point per chain at a time. For the
  reduced SIPNET problem that is 0.41 s per step for every chain in turn. The ideal: advance the
  chains in lockstep, evaluating each step's proposals, one per chain, as one batch, so the forward
  `Function`'s executor runs them together. Deferred by decision (§6).
- **PG-17 (new): one MCMC run's chains share one initial state.** `blackjax_rwmh` takes one `init`,
  or draws a single prior point, and starts every chain there (`_inference_utils.get_init_state`).
  Chains cannot start from EKI's particles, and R-hat cannot reveal a dependence on the start.
  (Minor: its `proposal_cov` option is documented both as a covariance and as a Cholesky factor.)
- **PG-18 (new): a `Function`'s declared traceability cannot be read back.** The reduced forward map
  must make the full one's claim; a stopgap reads the private attribute.
- **PG-09** (linear pushforward of a law) still decides the reduced model's ideal form, a law over
  $(a, b)$ pushed through $W$; the reparameterized forward `Function` works today, as design 01
  expected.

---

## 9. Decisions (approved 2026-11-06, as recommended)

1. **The toy problem** (§1): SIPNET's canopy equations in JAX with 14 calibration parameters, on the
   real data; a synthetic-twin option if you want a known truth.
2. **The reference target** (§2): the posterior-averaged Gauss–Newton information in whitened
   coordinates, with the prior average as the second reference.
3. **The shipped estimator** (§3.1): statistical linearizations accumulated over the EKI ladder,
   weighted by $\Delta\beta$; the others of §3.2 as tools for comparison.
4. **The rank rule** (§3.3): discarded eigenvalues at most ½, always overridable.
5. **The reduced posterior** (§4): the plug-in, sampled by ProbPipe's `blackjax_rwmh`; the optimal
   one only as a toy benchmark; pseudo-marginal not shipped.
6. **EKI with inferred scales** (§5): the Harvard Forest EKI-Gibbs with the common gain, built on
   EnsKit's step API.
7. **The algorithm** (§6): `active_subspace_mcmc` as the one shipped algorithm.
8. **Scope**: metrics, replicates and plots stay in design 04; this design's scores are the
   prototype's.

---

## Appendix: what the prototype verified

| Claim | Script | Result |
|---|---|---|
| The toy runs fast and resembles SIPNET | `toy.py` | 0.9 ms per run, 10 ms per Jacobian; GPP 1264, NEE −280; correlation 0.93 with the data |
| NUTS gives the toy's reference posterior | `t1_toy_nuts.py` | 4 × 1000 draws, 677 s (885 s inferred scales), R-hat ≤ 1.001 |
| One linearization estimates $\bar H$, not $H$ | `t2_estimators.py` | prior: 42.8 vs 43.2 (Stein); $H$ 53.0, 20.0, 11.2 vs $\bar H$ 43.2, 16.4, 4.1 |
| The ladder estimate's directions are near-optimal | `t3_compare.py` | 0.98–0.99 of the best information at r = 1–8, J = 50–200 |
| Its eigenvalues are biased low; the rank rule still lands at 7 | `t2`, `t3`, session check | leading 43 vs 77; rule gives r = 7 in 5 of 5 seeds |
| The plug-in is near-optimal; pseudo-marginal does not mix | `t4_reduced.py` | table of §4 |
| ProbPipe's RWMH samples the reduced problem and beats full RWMH | `t6_probpipe_reduced.py`, `t8_full_rwmh.py` | 0.143 / 0.056 vs 0.352 / 0.365 at the same budget |
| EKI-Gibbs recovers the noise scales | `t5_eki_gibbs.py` | intervals of §5 |
| The pipeline runs on SIPNET | `t7_sipnet_small.py` | §6 |

References: Constantine, Kent & Bui-Thanh (2016), *SISC* 38:A2779; Cui, Martin, Marzouk, Solonen &
Spantini (2014), *Inverse Problems* 30:114015; Zahm, Cui, Law, Spantini & Marzouk (2022), *Math.
Comp.* 91:1789; Polette et al. (2026), arXiv:2604.13563; Botha et al. (2023), *Inverse Problems*
39:125014; Doucet, Pitt, Deligiannidis & Kohn (2015), *Biometrika* 102:295.
