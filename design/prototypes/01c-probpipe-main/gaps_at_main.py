"""Re-check every open entry of the ProbPipe gap registry at the pinned ProbPipe (main, 9158a50e).

Run from the repository root, in the repository's environment:

    uv run python -W ignore design/prototypes/01c-probpipe-main/gaps_at_main.py

Each check prints one line saying what ProbPipe does. Lines starting with "still open" describe
behaviour the registry records as a gap; "fixed" lines describe behaviour that closed one.
"""

import logging
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

logging.disable(logging.WARNING)  # PG-14's provenance warning would flood the output

from probpipe import (  # noqa: E402  (after JAX, before EnsKit: EK-01 needs 32-bit first)
    Distribution,
    EmpiricalDistribution,
    EvaluationError,
    Function,
    LogNormal,
    Normal,
    NumericArrayBatch,
    NumericArraySpec,
    NumericRecordSpec,
    OutputSpec,
    ResolutionError,
    SupportsLogProb,
    bijector_for,
    condition_on,
    conditional_distribution,
    inverse,
    log_prob,
    positive,
    quantile,
    sample,
    workflow_run,
)
from probpipe.distributions import ConditionalDistribution  # noqa: E402


def report(gap, text):
    print(f"{gap}: {text}")


def first_line(error):
    return f"{type(error).__name__}: {str(error).splitlines()[0][:150]}"


# ---------------------------------------------------------------- PG-01, fixed
def opaque_square(theta):
    return np.asarray([float(np.asarray(theta)) ** 2], dtype="float32")


opaque = Function(
    opaque_square, traceable=False, output_spec=NumericArraySpec((1,), dtype="float32")
)
kernel = conditional_distribution(
    lambda theta: Normal("y", opaque(theta), 0.1), given_spec={"theta": NumericArraySpec(())}
)
with workflow_run(seed=0):
    draws = sample(kernel & Normal("theta", 0.0, 1.0), 4)
report("PG-01", f"fixed: a joint around an opaque Function samples, y shape {draws['y'].raw.shape}")

# ---------------------------------------------------------------- PG-03
try:
    condition_on.using(learning_rate={"y": 0.5})
    report("PG-03", "fixed: condition_on takes a learning rate")
except TypeError as error:
    report("PG-03", f"still open: {first_line(error)}")

# ---------------------------------------------------------------- PG-05
bijector = bijector_for(positive)
try:
    inverse(bijector)
    report("PG-05", "fixed: inverse(bijector) works")
except NotImplementedError as error:
    report("PG-05", f"still open: inverse(bijector) raises {first_line(error)}")
grouped = (LogNormal("k", 0.0, 1.0) & Normal("m", 0.0, 1.0)).rename(
    {"k": "theta/k", "m": "theta/m"}
) & LogNormal("s2", 0.0, 1.0)
with workflow_run(seed=0):
    view_draws = sample(grouped["theta"], 4)
report("PG-05", f"public: draws of a view lay out as vectors, shape {view_draws.to_vector().shape}")

# ---------------------------------------------------------------- PG-06, PG-07, PG-10
CALLS = {"count": 0, "active": 0, "most_active": 0}
LOCK = threading.Lock()


def slow_body(x):
    with LOCK:
        CALLS["count"] += 1
        CALLS["active"] += 1
        CALLS["most_active"] = max(CALLS["most_active"], CALLS["active"])
    value = float(np.asarray(x))
    time.sleep(0.05)
    with LOCK:
        CALLS["active"] -= 1
    if value > 100:
        raise RuntimeError(f"solver failed at {value}")
    return np.asarray([value, value**2], dtype="float32")


def counted(thunk):
    CALLS.update(count=0, most_active=0)
    try:
        thunk()
    except EvaluationError as error:
        return CALLS["count"], error
    return CALLS["count"], None


spec = NumericArraySpec((2,), dtype="float32")
points = NumericArrayBatch(jnp.arange(8.0, dtype=jnp.float32), "x", label="xs")
probed = Function(slow_body, output_spec=spec)
calls, _ = counted(lambda: probed(points))
again, _ = counted(lambda: probed(points))
report("PG-10", f"traceable=None runs the body {calls} times for 8 points, then {again}")
declared = Function(slow_body, traceable=False, output_spec=spec)
calls, _ = counted(lambda: declared(points))
repeated, _ = counted(lambda: declared(points))
report("PG-06", f"still open: an identical call re-runs the body ({calls} then {repeated} runs)")
failing = NumericArrayBatch(jnp.asarray([0, 1, 200, 3], dtype=jnp.float32), "x", label="xs")
_, error = counted(lambda: declared(failing))
kept = sum(output is not None for output in error.outputs)
failed = [failure.point_index for failure in error.failures]
report("PG-07", f"public: EvaluationError names failed points {failed} and keeps {kept} outputs")
atoms = EmpiricalDistribution(jnp.arange(300.0) / 10, output_name="x")
with workflow_run(seed=0):
    lifted = Function(lambda x: x * 2.0, traceable=True)(atoms)
report("PG-10", f"still open: lifting over 300 atoms gives {lifted.atoms.raw.shape[0]}, silently")

theta_kernel = conditional_distribution(
    lambda theta: Normal("y", Function(
        slow_body, traceable=False, output_spec=spec, label="slow"
    )(theta), 0.5),
    given_spec={"theta": NumericArraySpec(())},
)
CALLS.update(count=0, most_active=0)
with workflow_run(seed=0):
    condition_on.using(
        method="blackjax_rwmh",
        method_options={"num_chains": 2, "num_warmup": 10, "num_results": 20},
        executor="thread",
        executor_options={"max_workers": 8},
    )(theta_kernel & Normal("theta", 0.0, 1.0), {"y": jnp.asarray([0.5, 0.3])})
report(
    "PG-07",
    f"still open: RWMH with executor='thread' ran {CALLS['count']} evaluations, "
    f"at most {CALLS['most_active']} at a time",
)

# ---------------------------------------------------------------- PG-08
try:
    Function(lambda x: x, differentiable=NumericArraySpec(()))
    report("PG-08", "fixed: Function accepts a differentiability claim")
except NotImplementedError as error:
    report("PG-08", f"still open: {first_line(error)}")


# ---------------------------------------------------------------- PG-11
class OnlyLogProb(Distribution):
    def __init__(self):
        super().__init__(OutputSpec(y=NumericArraySpec(())), label="only_log_prob")

    def _log_prob(self, value):
        return -0.5 * jnp.asarray(value) ** 2


report(
    "PG-11",
    "still open: a law defining only _log_prob satisfies SupportsLogProb: "
    f"{isinstance(OnlyLogProb(), SupportsLogProb)}",
)


# ---------------------------------------------------------------- PG-12
class TemperedKernel(ConditionalDistribution):
    def __init__(self):
        super().__init__(
            {"mu": NumericArraySpec(())}, OutputSpec(y=NumericArraySpec(())), label="tempered"
        )

    def _conditional_unnormalized_log_prob(self, given, value):
        return 0.5 * (-0.5 * (jnp.asarray(value) - given["mu"]) ** 2)

    def _condition_on(self, given, /, **options):
        kernel = self

        class AtGiven(Distribution):
            def __init__(self):
                super().__init__(kernel.event_spec, label="at given")

            def _unnormalized_log_prob(self, value):
                return kernel._conditional_unnormalized_log_prob(given, value)

        return AtGiven()


with workflow_run(seed=0):
    bound = condition_on.using(method_options={"num_warmup": 50, "num_results": 50})(
        TemperedKernel(), {"mu": 0.0}
    )
report(
    "PG-12",
    f"still open: binding an unnormalized kernel's given runs inference ({type(bound).__name__})",
)

# ---------------------------------------------------------------- PG-13
theta = (Normal("a", 0.0, 1.0) & Normal("b", 0.0, 2.0)).rename({"a": "theta/a", "b": "theta/b"})
likelihood = conditional_distribution(
    lambda theta: Normal("y", theta["a"] + theta["b"] * jnp.arange(3.0), 0.5),
    given_spec={"theta": NumericRecordSpec(a=NumericArraySpec(()), b=NumericArraySpec(()))},
)
(prior_factor,) = [f for f in (likelihood & theta).factors if "theta" in f.event_spec.outputs]
value = {"theta": {"a": jnp.asarray(0.3), "b": jnp.asarray(-0.4)}}
try:
    log_prob(prior_factor, value)
    report("PG-13", "fixed: the factor taken from the joint scores grouped values")
except Exception as error:  # noqa: BLE001
    report("PG-13", f"still open: {first_line(error)}")
report("PG-13", f"its draws have the fields {sorted(prior_factor._sample(jax.random.key(0)))}")

# ---------------------------------------------------------------- PG-14
from probpipe.core._fingerprint import fingerprint  # noqa: E402

operator = sp.random(5, 40, density=0.2, format="csr", random_state=0)


def closure_over(matrix):
    def body(x):
        return jnp.asarray(matrix @ np.full(40, float(np.asarray(x))))

    return body


try:
    fingerprint(Function(closure_over(operator), traceable=False, label="closure"))
    report("PG-14", "fixed: a closure over a sparse matrix fingerprints")
except RecursionError:
    report("PG-14", "still open: fingerprinting a closure over a sparse matrix recurses")


class Holder:
    def __init__(self, matrix):
        self.matrix = matrix

    def apply(self, x):
        return jnp.asarray(self.matrix @ np.full(40, float(np.asarray(x))))


doubled = operator.copy()
doubled.data *= 2
same = fingerprint(Function(Holder(operator).apply, traceable=False, label="g")) == fingerprint(
    Function(Holder(doubled).apply, traceable=False, label="g")
)
report("PG-14", f"bound methods over different operators fingerprint alike: {same}")

# ---------------------------------------------------------------- PG-15
law = Normal("x", 0.0, 1.0)
try:
    sample.using(key=jax.random.key(0))
    report("PG-15", "fixed: sample takes a key")
except TypeError as error:
    report("PG-15", f"still open: {first_line(error)}")
with workflow_run(seed=1):
    plain = np.asarray(sample(law, 3).raw)
with workflow_run(seed=1):
    with workflow_run(seed=5):
        sample(law, 2)
    after_nested = np.asarray(sample(law, 3).raw)
report("PG-15", f"a nested workflow_run leaves the outer stream unchanged: "
       f"{np.array_equal(plain, after_nested)}")

# ---------------------------------------------------------------- EK-01
model = conditional_distribution(
    lambda theta: Normal("y", theta * jnp.ones(3), 0.5), given_spec={"theta": NumericArraySpec(())}
) & Normal("theta", 0.0, 1.0)  # built with 32-bit floats
import enskit  # noqa: E402, F401  (switches JAX to 64-bit floats)

try:
    with workflow_run(seed=0):
        condition_on.using(
            method="blackjax_rwmh",
            method_options={"num_chains": 1, "num_warmup": 20, "num_results": 20},
        )(model, {"y": jnp.asarray([0.2, 0.4, 0.3])})
    report("EK-01", "fixed: a law built before importing EnsKit conditions after it")
except TypeError as error:
    report("EK-01", f"still open: {first_line(error)}")


# ---------------------------------------------------------------- PG-08, method selection
# After EK-01, since lsmas imports EnsKit. The likelihood is JAX code around an opaque forward map.
from lsmas._stopgaps.likelihood import GaussianSimulatorLikelihood  # noqa: E402


def opaque_linear(theta):
    a, b = float(np.asarray(theta["a"])), float(np.asarray(theta["b"]))
    return np.asarray([a + b, a - b], dtype="float64")


opaque_forward = Function(
    opaque_linear, traceable=False, output_spec=NumericArraySpec((2,), jnp.float64), label="opaque"
)
simulator_likelihood = GaussianSimulatorLikelihood(
    opaque_forward,
    output_name="y",
    theta_spec=NumericRecordSpec(a=NumericArraySpec(()), b=NumericArraySpec(())),
    reference_variance=[1.0, 1.0],
    classes=["day", "day"],
    scales={"day": 1.0},
)
opaque_model = simulator_likelihood & (Normal("a", 0.0, 1.0) & Normal("b", 0.0, 1.0)).rename(
    {"a": "theta/a", "b": "theta/b"}
)
observed = {"y": jnp.asarray([0.5, 0.3])}
selected = condition_on.check(opaque_model, observed).method
try:
    with workflow_run(seed=0):
        condition_on.using(method_options={"num_warmup": 5, "num_results": 5})(
            opaque_model, observed
        )
    report("PG-08", f"fixed: automatic selection picks {selected}, which runs")
except TypeError as error:
    report(
        "PG-08",
        f"still open: automatic selection picks {selected}, which fails: {first_line(error)}",
    )


# ---------------------------------------------------------------- PG-16, transformed quantiles
from lsmas.priors import logit_normal_from_interval  # noqa: E402

logit_normal = logit_normal_from_interval("share", 0.04, 0.20)
exact = float(logit_normal.bijector(quantile(logit_normal.base, 0.5)).raw)
try:
    stated = float(quantile.using(exact_only=True)(logit_normal, 0.5).raw)
    report("PG-16", f"fixed: the exact median is stated, {stated:.4f} (exact {exact:.4f})")
except ResolutionError as error:
    with workflow_run(seed=0):
        estimate = float(quantile(logit_normal, 0.5).raw)
    report(
        "PG-16",
        f"still open: {estimate:.4f} by Monte Carlo against the exact {exact:.4f}; "
        f"under exact_only, {first_line(error)}",
    )
