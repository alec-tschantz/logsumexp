from jax import numpy as jnp, tree_map, grad


def solve(energy_cfg, variables, steps=10, lr=1e-2):
    for _ in range(steps):
        variables = grad_step(energy_cfg, variables, lr)
    return variables


def grad_step(energy_cfg, variables, lr=1e-2):
    def loss_fn(vars_):
        return energy(energy_cfg, vars_)

    grads = grad(loss_fn)(variables)
    return tree_map(lambda x, g: x - lr * g, variables, grads)


def energy(energy_cfg, variables):
    total = 0.0
    for energy_mod, varnames in energy_cfg:
        args = [variables[vn] for vn in varnames]
        total = total + energy_mod(*args)
    return total
