import equinox as eqx
from jax import Array, nn, numpy as jnp, random as jr
from typing import Any


class Energy(eqx.Module):
    def measure(self, *args: Any) -> Array:
        raise NotImplementedError

    def __call__(self, *args: Any) -> float:
        m = self.measure(*args)
        return -jnp.sum(nn.logsumexp(m, axis=1))


class CrossAttention(Energy):
    Wk: Array
    Wq: Array

    def __init__(self, D: int, key: Array) -> None:
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

    def __init__(self, D: int, key: Array) -> None:
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


class Mixture(Energy):
    Sigma: Array

    def __init__(self, D: int) -> None:
        self.Sigma = jnp.eye(D)

    def measure(self, x: Array, mu: Array) -> Array:
        diffs = x[:, None, :] - mu[None, :, :]
        inv = jnp.linalg.inv(self.Sigma)
        dist = 0.5 * jnp.einsum("nkd,dd,nkd->nk", diffs, inv, diffs)
        logdet = 0.5 * jnp.log(jnp.linalg.det(self.Sigma))
        c = 0.5 * x.shape[-1] * jnp.log(2.0 * jnp.pi)
        return -(dist + logdet + c)
