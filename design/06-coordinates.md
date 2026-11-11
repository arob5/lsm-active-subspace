# Design 06: coordinates as ProbPipe maps

Status: **implemented** (2026-11-11); decisions 1–4 approved as recommended. Prototype: `design/prototypes/06-coordinates/t1_maps.py`.

## 1. What is wrong with the current interface

`problem.coordinates` (`lsmas._stopgaps.coordinates`) works on NumPy arrays of shape $(\ldots, d)$,
not on ProbPipe values:

- **Draws must be flattened first.** Every map takes an array, so a user calls
  `array_from_draws(draws)` and must remember that the columns follow `coordinates.names`. ProbPipe
  already does this: `draws["theta"].to_vector()`, with `NumericRecordBatch.from_vector` as its
  inverse, in the same order.
- **One idea, eleven public names, two classes.** `bijection.to_unconstrained`, `to_constrained`,
  `record_from_unconstrained` and `record_batch`; `whiten`, `unwhiten`, `whitened_from_parameters`,
  `parameters_from_whitened`, `parameter_record` and `array_from_draws`; and the module function
  `block_values`. Several are compositions of others, and they return arrays, dicts or batches
  depending on the method.
- **Inconsistent names.** `whiten` maps $z \mapsto \xi$, but `whitened_from_parameters` maps
  $\theta \mapsto \xi$.
- **Not ProbPipe values.** A posterior cannot be mapped to whitened coordinates as a distribution,
  and the results cannot go to `plot_marginals`, `summarize` or the forward model without
  rebuilding records. `InverseProblem.predicted_observations` exists only to bridge arrays back to
  the forward model.

## 2. Proposal

**A change of coordinates is an invertible ProbPipe `Function` on records.**

```python
coordinates = problem.coordinates
coordinates.unconstrain        # θ ↦ z, a Function of the record theta
coordinates.whiten             # θ ↦ ξ
inverse(coordinates.whiten)    # ξ ↦ θ (ProbPipe's inverse operation)
```

- Each map takes the record `theta` and returns a record with the same field names. For $z$ that
  is exact, since $z_j = b_j^{-1}(\theta_j)$. For $\xi$ the names label the whitened coordinates,
  as `Subspace.parameter_names` and the reference diagnostics already do. ($L$ is lower-triangular,
  so $\xi_j$ depends on $\theta_1, \dots, \theta_j$ only.)
- The maps claim `SupportsInverse` and `SupportsLogDetJacobian`, ProbPipe's contract for a bijector:
  the same one `bijector_for` and `BijectorTransformedDistribution` use. They are JAX code
  (`traceable=True`), so batches vectorize and the maps can be differentiated.
- Because they are `Function`s, they take whatever ProbPipe passes a `Function`:

| Task | Now | Proposed |
|---|---|---|
| draws of θ as an array | `coordinates.array_from_draws(draws)` | `draws["theta"].to_vector()` |
| prior draws in ξ | `coordinates.whiten(coordinates.bijection.to_unconstrained(coordinates.array_from_draws(draws)))` | `coordinates.whiten(draws["theta"])` |
| the posterior in ξ, as a distribution | not available | `coordinates.whiten(posterior["theta"])` |
| the prior in ξ | not available | `pushforward(prior["theta"], coordinates.whiten)` |
| back from ξ to θ | `coordinates.parameters_from_whitened(xi)` | `inverse(coordinates.whiten)(xi)` |
| $G$ at points in ξ | `problem.predicted_observations(coordinates.parameters_from_whitened(xi))` | `problem.forward(inverse(coordinates.whiten)(xi))` |
| an array of ξ as records | not available | `NumericRecordBatch.from_vector(coordinates.spec, array, level_names="draw")` |
| plot or summarize | not available | `plot_marginals(coordinates.whiten(posterior["theta"]))` |

**Public surface of `Coordinates`:** `from_prior`, `names`, `dimension`, `spec` (the record spec of
`theta`), `unconstrain`, `whiten`, and $m$ and $L$ as `whitening_mean` and `whitening_factor`,
which the tests' closed-form information matrices read. Nothing else.

**Private:** the array kernels the algorithms run on, such as EKI's particles, MCMC states and
reference chains, which hold $(n, d)$ NumPy arrays and do linear algebra on them. These become
private methods of `Coordinates` that the public maps are built from, so there is one
implementation. `ParameterBijection` becomes a private helper. `block_values` stays private while
PG-13 lasts.

**Removed:** `array_from_draws`, `bijection`, `record_from_unconstrained`, `record_batch`, the
array `whiten` and `unwhiten`, `whitened_from_parameters`, `parameters_from_whitened`,
`parameter_record`, and `InverseProblem.predicted_observations`.

**Where it lives.** `Coordinates` moves to a public module, `lsmas.coordinates`, since the whitening
is ours whatever ProbPipe provides. What remains a stopgap is small and tagged:

- `STOPGAP(PG-05)`: ProbPipe's `inverse(f)` and `log_det_jacobian(f, x)` are declared but raise
  `NotImplementedError`, and a support bijector's inverse is reachable only through its private
  `_inverse`. `lsmas._stopgaps.inverse.inverse(f)` has ProbPipe's signature and calls the map's
  `_inverse`. Once ProbPipe implements the operation, the import changes to `from probpipe import
  inverse` and nothing else changes, since our maps already implement ProbPipe's `_inverse` and
  `_log_det_jacobian`. PG-05's entry is updated to say this.

## 3. Verified by the prototype (toy problem, 14 parameters)

- `whiten` maps a record, a batch of 500 draws (0.19 s with compilation), an
  `EmpiricalDistribution` and, through `pushforward`, the prior; `plot_marginals` draws the result.
- `inverse(whiten)` maps the batch back to θ exactly, and the forward model takes the result.
- `jax.jacfwd` through `inverse(whiten).using(raw=True)` gives the forward map's Jacobian in
  whitened coordinates, which `exact_subspace` needs.
- `to_vector` and `NumericRecordBatch.from_vector` give the arrays and records back.

## 4. Work

1. `lsmas.coordinates` with the new `Coordinates`; `lsmas._stopgaps.inverse`; remove
   `lsmas._stopgaps.coordinates`.
2. Callers: `InverseProblem`, `nee_likelihood`, `tools.reduced`, `tools.estimators`, `inference.enskit_eki`,
   `algorithms.active_subspace_mcmc`, `experiments.metrics`, `experiments.study`,
   `experiments.reference` and `plots`, each using the public maps on ProbPipe values and the
   private kernels on arrays.
3. Tests: `test_coordinates` rewritten around the maps (round trips, a law, a batch, the
   Jacobian, prior moments of ξ); the other tests follow the callers.
4. The notebook's section 8 shows the maps on the prior's draws and on the prior itself, with
   `plot_marginals` in place of the hand-written histograms helper.
5. `docs/notation.md` names the maps next to $z$ and $\xi$.

## 5. Decisions

1. **Records, not vectors.** Each map returns a record with θ's field names, so its result plots,
   summarizes and goes to `to_vector()` like any draw. The alternative is a vector of shape $(d,)$,
   closer to the mathematics of $\xi$ but without names. Recommended: records.
2. **Two maps, both from θ.** `unconstrain` (θ ↦ z) and `whiten` (θ ↦ ξ); $z \mapsto \xi$ is not
   exposed, since users start from θ. Recommended.
3. **The inverse through `inverse(f)`,** ProbPipe's operation, rather than separate `constrain` and
   `unwhiten` maps: one name per map, as ProbPipe intends. Recommended.
4. **`predicted_observations` removed**, since `problem.forward(inverse(coordinates.whiten)(xi))`
   does its job on ProbPipe values. Recommended.

## 6. Also: two renames in `lsmas.parameters`

- `_SIPNET_PARAMETERS_CALIBRATED_AS_THEY_ARE` → `_DIRECT_SIPNET_PARAMETERS`, with the comment
  "Calibration parameters that are SIPNET parameters, under the same name and in the same units."
  Alternatives: `_UNCHANGED_SIPNET_PARAMETERS`, `_SHARED_NAME_PARAMETERS`.
- `_INITIAL_CONDITIONS` → `_IC_CALIBRATION_TO_SIPNET_NAME`, as proposed: the initial conditions
  that are SIPNET parameters under another name, by calibration name.
