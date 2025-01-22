# logsumexp

```bash
pip install -e .
```

```python
from jax import numpy as jnp, random as jr
from logsumexp import energy, solve

key = jr.PRNGKey(0)
D, M, K, N = 4, 3, 10, 32

x = jnp.zeros((N, D))
z = jnp.zeros((K, D))
m = jnp.zeros((M, D))

hopfield = (energy.Hopfield(), ["z", "m"])
slot = (energy.SlotAttention(D, key), ["x", "z"])
variables = {"x": x, "z": z, "m": m}

variables = solve([slot, hopfield], variables)
```