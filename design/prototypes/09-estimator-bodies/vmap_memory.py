"""Time and peak memory of the toy model's whitened Jacobian over N points, in pure JAX.

Run as ``uv run python design/prototypes/09-estimator-bodies/vmap_memory.py <mode> <N>``, with
<mode> one of vmap_eager (as ProbPipe evaluates today), vmap_jit (compiled, all points at once)
and chunked_jit (compiled, 25 points at a time with jax.lax.map). Run each in a fresh process:
the peak memory is the process maximum.
"""

import resource
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from lsmas.coordinates import inverse
from lsmas.problems import load_inverse_problem

mode, n = sys.argv[1], int(sys.argv[2])
toy = load_inverse_problem("toy")
coordinates = toy.coordinates
names = coordinates.names
forward = toy.forward.using(raw=True)
to_theta = inverse(coordinates.whiten).using(raw=True)


def jacobian_at(z):
    return jax.jacfwd(lambda v: forward(to_theta(dict(zip(names, v, strict=True)))))(z)


z = jnp.asarray(np.random.default_rng(0).normal(size=(n, len(names))))
if mode == "vmap_eager":
    run = jax.vmap(jacobian_at)
elif mode == "vmap_jit":
    run = jax.jit(jax.vmap(jacobian_at))
elif mode == "chunked_jit":
    run = jax.jit(lambda points: jax.lax.map(jacobian_at, points, batch_size=25))
# The timed call includes compilation for the jit modes (well under a second here).
start = time.time()
out = run(z).block_until_ready()
elapsed = time.time() - start
peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
print(f"{mode:12s} N={n:4d}: {elapsed:6.1f} s, peak memory {peak_gb:5.1f} GB, output {out.shape}")
