"""Design 10: a Gaussian simulator likelihood whose noise scales are an optional input.

Checks, at the pinned ProbPipe, that one likelihood serves fixed and inferred scales, that it
agrees with the current likelihood, and that the noise variance at a draw is the variance of the
likelihood's law there. Run with
``uv run python design/prototypes/10-model-and-inference/optional_scales.py``.
"""

import time

import numpy as np
from probpipe import (
    EmpiricalDistribution,
    condition_on,
    expectation,
    sample,
    variance,
    workflow_run,
)
from scaled_noise_likelihood import ScaledNoiseLikelihood

from lsmas._stopgaps.tempering import tempered_condition_on
from lsmas.problems import load_inverse_problem


def report(check, outcome):
    print(f"{check:<66} {outcome}")


def outcome_of(call):
    try:
        return call()
    except Exception as error:
        return f"{type(error).__name__}: {str(error)[:400]}"


workflow_run.set_default(seed=0)
current = {scales: load_inverse_problem("toy", scales=scales) for scales in ("fixed", "inferred")}
old = current["inferred"].likelihood
likelihood = ScaledNoiseLikelihood(
    old.forward,
    output_name="nee",
    reference_variance=old.reference_variance,
    classes=old.classes,
    default_scales={"day": 1.0, "night": 1.0},
    label="nee_likelihood",
)
report("1. given spec", likelihood.given_spec)

priors = {scales: current[scales].prior for scales in current}
models = {scales: likelihood & priors[scales] for scales in priors}
for scales, model in models.items():
    report(
        f"2. fields of a draw of likelihood & prior, {scales}",
        outcome_of(lambda m=model: list(sample(m))),
    )

draws = {scales: sample(priors[scales], 4) for scales in priors}
observed = current["fixed"].data["nee"]
for scales in priors:
    draw = draws[scales][0]
    given = {"theta": draw["theta"]} | ({"s2": draw["s2"]} if scales == "inferred" else {})
    new = likelihood.pointwise_log_prob(given, observed)
    old_given = given if scales == "inferred" else {"theta": draw["theta"]}
    old_value = current[scales].likelihood.pointwise_log_prob(old_given, observed)
    report(
        f"3. pointwise log-density matches the current likelihood, {scales}",
        bool(np.allclose(new, old_value)),
    )

for scales in priors:
    draw = draws[scales][0]
    got = outcome_of(lambda d=draw: np.asarray(variance(condition_on(likelihood, d)).raw))
    expected = np.asarray(
        current[scales].likelihood.noise_variances(
            {c: draw["s2"][c] for c in ("day", "night")} if scales == "inferred" else None
        )
    )
    report(
        f"4. variance(condition_on(likelihood, draw)), {scales}",
        got if isinstance(got, str) else bool(np.allclose(got.reshape(-1), expected)),
    )

for scales in priors:
    held = EmpiricalDistribution(draws[scales])
    report(
        f"5. variance(condition_on(likelihood, draw)) in an expectation, held, {scales}",
        outcome_of(
            lambda h=held: (
                np.asarray(
                    expectation(h, lambda d: variance(condition_on(likelihood, d))).raw
                ).shape
            )
        ),
    )
    report(
        f"   the same over the prior, {scales}",
        outcome_of(
            lambda s=scales: (
                np.asarray(
                    expectation.using(num_samples=4)(
                        priors[s], lambda d: variance(condition_on(likelihood, d))
                    ).raw
                ).shape
            )
        ),
    )


for scales in priors:
    for method, options in [
        ("blackjax_nuts", {"num_chains": 1, "num_warmup": 30, "num_results": 20}),
        ("blackjax_rwmh", {"num_chains": 1, "num_warmup": 30, "num_results": 20}),
    ]:

        def run(s=scales, m=method, o=options):
            posterior = tempered_condition_on(
                models[s],
                current[s].data,
                learning_rate=current[s].learning_rate,
                method=m,
                method_options=o,
            )
            return f"draws with fields {list(posterior.atoms)}"

        report(f"6. condition likelihood & prior, {scales}, {method}", outcome_of(run))

small = load_inverse_problem("harvard_forest_small")
small_likelihood = ScaledNoiseLikelihood(
    small.forward,
    output_name="nee",
    reference_variance=small.likelihood.reference_variance,
    classes=small.likelihood.classes,
    default_scales={"day": 1.0, "night": 1.0},
    label="nee_likelihood",
)
small_draw = sample(small.prior)
start = time.time()
small.forward(small_draw["theta"])
forward_seconds = time.time() - start
start = time.time()
variance(condition_on(small_likelihood, small_draw))
report(
    "7. SIPNET: one forward run vs variance at a draw",
    f"{forward_seconds:.3f} s vs {time.time() - start:.3f} s",
)

for scales in priors:
    draw = draws[scales][0]
    by_condition = np.asarray(variance(condition_on(likelihood, draw)).raw)
    report(
        f"8. variance(likelihood, given=draw) is the same, {scales}",
        outcome_of(
            lambda d=draw, expected=by_condition: bool(
                np.allclose(np.asarray(variance(likelihood, given=d).raw), expected)
            )
        ),
    )
