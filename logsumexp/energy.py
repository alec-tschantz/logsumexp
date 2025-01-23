from typing import Any, Callable, List, Tuple, Dict

import equinox as eqx
from jax import Array, nn, numpy as jnp, random as jr, jit


class Energy(eqx.Module):
    def measure(self, *args: Any) -> Array:
        raise NotImplementedError

    def __call__(self, *args: Any) -> float:
        m = self.measure(*args)
        return -jnp.sum(nn.logsumexp(m, axis=1))


@eqx.filter_jit
def energy(edges, nodes) -> float:
    total = 0.0
    for energy_fn, names in edges:
        args = [nodes[name] for name in names]
        total = total + energy_fn(*args)
    return total


class CrossAttention(Energy):
    Wk: Array
    Wq: Array

    def __init__(self, D: int, key: jr.PRNGKey):
        k1, k2 = jr.split(key, 2)
        self.Wk = jr.normal(k1, (D, D))
        self.Wq = jr.normal(k2, (D, D))

    def measure(self, xq: Array, xk: Array) -> Array:
        Q = xq @ self.Wq
        K = xk @ self.Wk
        return Q @ K.T


class Hopfield(Energy):
    def measure(self, x: Array, m: Array) -> Array:
        return x @ m.T


class SlotAttention(Energy):
    Wk: Array
    Wq: Array

    def __init__(self, D: int, key: jr.PRNGKey):
        k1, k2 = jr.split(key, 2)
        self.Wk = jr.normal(k1, (D, D))
        self.Wq = jr.normal(k2, (D, D))

    def measure(self, x: Array, slots: Array) -> Array:
        K = x @ self.Wk
        Q = slots @ self.Wq
        return K @ Q.T


class SelfAttention(Energy):
    Wk: Array
    Wq: Array

    def __init__(self, D: int, key: jr.PRNGKey) -> None:
        k1, k2 = jr.split(key, 2)
        self.Wk = jr.normal(k1, (D, D))
        self.Wq = jr.normal(k2, (D, D))

    def measure(self, x: Array) -> Array:
        Q = x @ self.Wq
        K = x @ self.Wk
        return Q @ K.T


class GaussianMixture(Energy):
    Sigma: Array

    def __init__(self, D: int) -> None:
        self.Sigma = jnp.eye(D)

    def measure(self, x: Array, mu: Array) -> Array:
        diffs = x[:, None, :] - mu[None, :, :]
        inv = jnp.linalg.inv(self.Sigma)
        _, logdet = jnp.linalg.slogdet(self.Sigma)
        dist = jnp.einsum("nkd,dd,nkd->nk", diffs, inv, diffs)
        const = x.shape[-1] * jnp.log(2.0 * jnp.pi)
        return -0.5 * (dist + logdet + const)


class PredictiveCoding(Energy):
    W: Array
    b: Array
    func: Callable

    def __init__(self, D_in: int, D_out: int, func: Callable, key: jr.PRNGKey) -> None:
        k1, k2 = jr.split(key, 2)
        self.W = jr.normal(k1, (D_out, D_in))
        self.b = jr.normal(k2, (D_out,))
        self.func = func

    def measure(self, x: Array, mu: Array) -> Array:
        f_mu = self.func(mu @ self.W.T + self.b)
        diffs = x[:, None, :] - f_mu[None, :, :]
        return -0.5 * jnp.sum(diffs**2, axis=-1)


class LayerNorm(eqx.Module):
    gamma: Array
    delta: Array
    epsilon: float

    def __init__(self, D: int, epsilon: float = 1e-5, key=None) -> None:
        self.gamma = jnp.ones((D,))
        self.delta = jnp.zeros((D,))
        self.epsilon = epsilon

    def __call__(self, x: Array) -> float:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        norm = jnp.sqrt(var + self.epsilon)

        L = self.gamma * norm + jnp.sum(self.delta * x, axis=-1, keepdims=True)
        return jnp.sum(L)
