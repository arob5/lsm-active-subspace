# Checks for the move to ProbPipe `main`

Scratch code that backs the claims in [`../../probpipe-gaps.md`](../../probpipe-gaps.md) and
[`../../01c-probpipe-main-report.md`](../../01c-probpipe-main-report.md). It is not the repo's
code, and unlike the design-01 prototype it runs in the repository's own environment, which pins
ProbPipe `main` at `9158a50e`.

| File | What it checks |
|---|---|
| `gaps_at_main.py` | One line per behaviour the gap registry records: which gaps closed (PG-01), which are partly fixed (PG-05, PG-07, PG-10), and which are still open (PG-03, PG-06, PG-08, PG-11 to PG-15, EK-01) |

```bash
uv run python -W ignore design/prototypes/01c-probpipe-main/gaps_at_main.py
```

It takes about half a minute. The PG-08 selection check comes last because it imports `lsmas`,
which imports EnsKit and so would change the EK-01 check. The output at `9158a50e`:

```
PG-01: fixed: a joint around an opaque Function samples, y shape (4, 1)
PG-03: still open: TypeError: condition_on.using(): unknown control 'learning_rate'; …
PG-05: still open: inverse(bijector) raises NotImplementedError: inverse is not implemented yet
PG-05: public: draws of a view lay out as vectors, shape (4, 2)
PG-10: traceable=None runs the body 9 times for 8 points, then 8
PG-06: still open: an identical call re-runs the body (8 then 8 runs)
PG-07: public: EvaluationError names failed points [2] and keeps 3 outputs
PG-10: still open: lifting over 300 atoms gives 256, silently
PG-07: still open: RWMH with executor='thread' ran 67 evaluations, at most 2 at a time
PG-08: still open: NotImplementedError: Function(differentiable=...) is not supported yet; …
PG-11: still open: a law defining only _log_prob satisfies SupportsLogProb: False
PG-12: still open: binding an unnormalized kernel's given runs inference (EmpiricalDistribution)
PG-13: still open: ApplicabilityError: log_prob: … missing ['a', 'b'], unexpected ['theta']
PG-13: its draws have the fields ['a', 'b']
PG-14: still open: fingerprinting a closure over a sparse matrix recurses
PG-14: bound methods over different operators fingerprint alike: True
PG-15: still open: TypeError: sample.using(): unknown control 'key'; …
PG-15: a nested workflow_run leaves the outer stream unchanged: False
EK-01: still open: TypeError: cond branches must have equal output types but they differ.
PG-08: still open: automatic selection picks blackjax_nuts, which fails: TypeError: gradients cannot pass through 'opaque', …
```
