from typing import Dict, List, Tuple

from diffrax import ODETerm, Dopri5, PIDController, diffeqsolve
from jax import Array, numpy as jnp, grad, tree_map, tree_flatten, tree_unflatten

from logsumexp import Energy, energy


def solve_nodes(edges, nodes):
    flat_nodes, treedef = tree_flatten(nodes)

    def _ode_term(t, flat_nodes, edges):
        nodes = tree_unflatten(treedef, flat_nodes)
        grads = grad(lambda n: energy(edges, n))(nodes)
        return tree_map(lambda g: -g, tree_flatten(grads)[0])

    kwargs = {
        "t0": 0,
        "t1": 20,
        "dt0": None,
        "solver": Dopri5(),
        "terms": ODETerm(_ode_term),
        "stepsize_controller": PIDController(rtol=1e-3, atol=1e-3),
    }
    solution = diffeqsolve(y0=flat_nodes, args=edges, **kwargs)
    solution = tree_map(lambda y: y[-1], solution.ys)
    return tree_unflatten(treedef, solution)
