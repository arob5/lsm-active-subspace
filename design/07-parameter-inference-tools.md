# Design 07: the parameter inference tools notebook

Status: **implemented for review** (2026-11-12). The structure was agreed in conversation; the
notebook itself is the prototype.

## 1. Purpose

`notebooks/parameter_inference_tools.ipynb` introduces, on the toy problem, the tools the
algorithms are built from. The first notebook taught the inverse problem; this one teaches the
tools; the next one will teach running and summarizing experiments.

Conventions, from the reviewer's feedback on the first notebook:

- an overview first, plain titles, the full mathematics in a last section;
- each tool introduced with a short mathematical description and then the code that implements
  it, one tool per block;
- figures from `lsmas.plots`, or from helpers in a collapsed cell;
- ProbPipe idioms, with `to_vector()` only where linear algebra needs an array;
- the seed set once at the top;
- no exercises; side topics under "Additional details".

## 2. Outline

1. **Overview.** The posterior, why it is hard for SIPNET, the two kinds of tools (posterior
   approximation, parameter reduction), and the algorithm of section 7 as four steps.
2. **A Toy Inverse Problem.** What the toy is and why it exists; prior predictions against the data.
   *MCMC*: a short NUTS run (one chain, 100 draws). *The reference posterior*: the stored NUTS
   reference, prior against posterior marginals, posterior predictions.
3. **Ensemble Kalman inversion.** Tempering and the Kalman update; one call; EKI against the
   reference; the history of increments.
4. **Active subspaces.** The Jacobian in whitened coordinates; the information at a point; the
   information matrix over prior draws; the active subspace; choosing the rank; loadings;
   activity scores; active and inactive variables; which distribution to average over (prior
   against posterior, by spectra and principal angles).
5. **Subspaces without gradients.** Statistical linearization against the average Jacobian; the
   information from a linearization; linearizing along EKI (`LadderLinearization`); the
   sufficient summary plot. *Additional details*: a subspace from posterior draws
   (`covariance_subspace`).
6. **Reduced problems.** The reduced posterior; sampling it with random-walk Metropolis; lifting.
7. **Putting it together.** `active_subspace_mcmc`, its result, against the reference.
8. **Additional details: learning noise scales.** EKI's Gibbs draws of the scales, against the
   reference; what the estimators and the algorithm do with them.
9. **Mathematical details.** The information matrix and its linear-Gaussian meaning; statistical
   linearization and Stein's lemma; EKI's increments; the ladder estimate's weightings; the rank
   rule; the reduced problem and the lift; a table of symbols and code.

The notebook runs in about two and a half minutes; the NUTS demonstration and the two EKI runs
with their compilation take most of it.

## 3. Decisions (agreed 2026-11-12)

1. `active_subspace_mcmc` is introduced here, so that the notebook ends with the tools put
   together.
2. NUTS is demonstrated with a short run in "A Toy Inverse Problem", and the stored reference is
   used for every comparison.
3. Inferred noise scales get a short section near the end.
4. The notebook is `parameter_inference_tools.ipynb`, since not every tool is about parameter
   reduction.

## 4. Additions to the package

Each one replaces code that would otherwise sit in the notebook.

- **`lsmas.tools.jacobians(problem, z)`**: $DG(z)$ at one point or a batch, exactly, for a forward
  map JAX can differentiate. `exact_subspace` now computes its Jacobians with it, so the notebook
  can show the Jacobian, the information at a point and the average before handing all three to
  `exact_subspace`. The compiled function is cached per problem.
- **`Subspace.principal_angles(other)`** and **`lsmas.tools.principal_angles(basis, other)`**:
  the angles between two subspaces, in degrees. `metrics.largest_principal_angle` now uses the
  same function.
- **A short `Subspace` repr**: the estimator, the rank and the leading eigenvalues, in place of
  the full arrays.
- **`plots.plot_spectra(..., labels=...)`**: coloured, labelled lines for a few estimators side by
  side; without labels it draws grey lines for many runs, as before. Eigenvalues below $10^{-6}$ of
  the largest are drawn at that floor, since the toy has one direction the data do not inform at
  all (a combination of the photosynthetic capacity, the light extinction coefficient and the
  leaf-on growth leaves its predictions unchanged).
- **`plots.plot_features` takes a distribution of predictions**, such as
  `toy.forward.using(max_evaluations=200)(posterior["theta"])`.
- **`Reference.posterior`**: the stored reference as a ProbPipe `EmpiricalDistribution` of records
  (`theta`, and `s2` when the scales are inferred), so it plots and samples like any posterior.

## 5. A fix found while writing the notebook

The whitening moments of `Coordinates` are computed when first needed (design 06). When that first
need came inside a JAX trace, as when `jacobians` compiles the map from whitened coordinates, the
prior draws were traced and the computation failed. The moments are now computed eagerly
(`jax.ensure_compile_time_eval`), with a test that first needs them inside `jax.jit`.

## 6. Also

- The notebook test executes every tutorial but not the personal copies `notebooks/my_*.ipynb`
  the README tells students to work in.
- The README lists the two tutorials in order.
