# logsumexp

```bash
pip install -e .
```

```python
from jax import numpy as jnp, random as jr
from logsumexp import Hopfield, SlotAttention, solve_nodes, energy


key = jr.PRNGKey(0)
D, M, K, N = 4, 3, 10, 32

x = jnp.zeros((N, D))
z = jnp.zeros((K, D))
m = jnp.zeros((M, D))

nodes = {"x": x, "z": z, "m": m}
edges = [(Hopfield(), ["z", "m"]), (SlotAttention(D, key), ["x", "z"])]

nodes = solve_nodes(edges, nodes)
final_energy = energy(edges, nodes)    
```