from jax import Array, numpy as jnp, grad, tree_map
from diffrax import ODETerm, Heun, PIDController, diffeqsolve

from logsumexp import energy


def solve(edges, nodes):
    def _term(_, nodes, edges):
        grads = grad(lambda node: energy(edges, node))(nodes)
        return tree_map(lambda g: -g, grads)

    kwargs = {
        "t0": 0,
        "t1": 20,
        "dt0": None,
        "solver": Heun(),
        "terms": ODETerm(_term),
        "stepsize_controller": PIDController(rtol=1e-3, atol=1e-3),
    }
    solution = diffeqsolve(y0=nodes, args=edges, **kwargs)
    return tree_map(lambda y: y[-1], solution.ys)
