from typing import Any, Callable

import equinox as eqx
from jax import Array, nn, numpy as jnp, random as jr


class Energy(eqx.Module):
    def measure(self, *args: Any) -> Array:
        raise NotImplementedError

    def __call__(self, *args: Any) -> float:
        m = self.measure(*args)
        return -jnp.sum(nn.logsumexp(m, axis=1))


class CrossAttention(Energy):
    Wk: Array
    Wq: Array

    def __init__(self, D: int, key: Array):
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

    def __init__(self, D: int, key: Array):
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

    def __init__(self, D: int, key: Array) -> None:
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

    def __init__(self, D: int, func: Callable, key: Array) -> None:
        k1, k2 = jr.split(key, 2)
        self.W = jr.normal(k1, (D, D))
        self.b = jr.normal(k2, (D,))
        self.func = func

    def measure(self, x: Array, mu: Array) -> Array:
        f_mu = self.func(mu @ self.W.T + self.b)
        diffs = x[:, None, :] - f_mu[None, :, :]
        return -0.5 * jnp.sum(diffs**2, axis=-1)
