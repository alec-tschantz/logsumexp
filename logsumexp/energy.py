import equinox as eqx
from jax import Array, numpy as jnp, nn, random


class Energy(eqx.Module):
    def __call__(self, *args) -> Array:
        raise NotImplementedError


class Mixture(Energy):
    logpi: Array
    mu: Array
    prec: Array

    def __init__(self, K: int, D: int, key: Array):
        k1, k2, _ = random.split(key, 3)
        self.logpi = random.normal(k1, (K,))
        self.mu = random.normal(k2, (K, D))
        self.prec = jnp.ones((D,))

    def measure(self, x: Array, mu: Array) -> Array:
        # x: (N, D), mu: (K, D)
        diffs = x[:, None, :] - mu[None, :, :]  # (N, K, D)
        return self.logpi[None, :] - 0.5 * jnp.sum(
            self.prec * diffs**2, axis=-1
        )  # (N, K)

    def __call__(self, x: Array) -> Array:
        # x: (N, D)
        m = self.measure(x, self.mu)  # (N, K)
        return -jnp.sum(nn.logsumexp(m, axis=1))


class SelfAttention(Energy):
    Wk: Array
    Wq: Array

    def __init__(self, D: int, key: Array):
        k1, k2 = random.split(key, 2)
        self.Wk = random.normal(k1, (D, D))
        self.Wq = random.normal(k2, (D, D))

    def measure(self, xk: Array, xq: Array) -> Array:
        # xk: (N, D), xq: (N, D)
        return xk @ self.Wk @ (xq @ self.Wq).T

    def __call__(self, x: Array) -> Array:
        # x: (N, D)
        m = self.measure(x, x)  # (N, N)
        return -jnp.sum(nn.logsumexp(m, axis=0))


class CrossAttention(Energy):
    Wk: Array
    Wq: Array

    def __init__(self, D: int, key: Array):
        k1, k2 = random.split(key, 2)
        self.Wk = random.normal(k1, (D, D))
        self.Wq = random.normal(k2, (D, D))

    def measure(self, xk: Array, xq: Array) -> Array:
        # xk: (Nk, D), xq: (Nq, D)
        return xk @ self.Wk @ (xq @ self.Wq).T

    def __call__(self, xk: Array, xq: Array) -> Array:
        # xk: (Nk, D), xq: (Nq, D)
        m = self.measure(xk, xq)  # (Nk, Nq)
        return -jnp.sum(nn.logsumexp(m, axis=0))


class Hopfield(Energy):
    def measure(self, x: Array, mem: Array) -> Array:
        # x: (N, D), mem: (M, D)
        return x @ mem.T

    def __call__(self, x: Array, mem: Array) -> Array:
        # x: (N, D), mem: (M, D)
        m = self.measure(x, mem)  # (N, M)
        return -jnp.sum(nn.logsumexp(m, axis=1))


class SlotAttention(Energy):
    Wk: Array
    Wq: Array

    def __init__(self, D: int, key: Array):
        k1, k2 = random.split(key, 2)
        self.Wk = random.normal(k1, (D, D))
        self.Wq = random.normal(k2, (D, D))

    def measure(self, x: Array, slots: Array) -> Array:
        # x: (N, D), slots: (S, D)
        k = x @ self.Wk  # (N, D)
        q = slots @ self.Wq  # (S, D)
        return k @ q.T  # (N, S)

    def __call__(self, x: Array, slots: Array) -> Array:
        # x: (N, D), slots: (S, D)
        m = self.measure(x, slots)  # (N, S)
        return -jnp.sum(nn.logsumexp(m, axis=1))
