import equinox as eqx
from jax import Array, numpy as jnp, random as jr, nn


class Energy(eqx.Module):

    def measure(self, *args) -> Array:
        raise NotImplementedError

    def __call__(self, *args) -> float:
        m = self.measure(*args)
        return -jnp.sum(nn.logsumexp(m, axis=1))


class Mixture(Energy):
    mu: Array  # (K, D)
    prec: Array  # (D,)

    def __init__(self, K: int, D: int, key: Array):
        k1, _ = jr.split(key, 2)
        self.mu = jr.normal(k1, (K, D))
        self.prec = jnp.ones((D,))

    def measure(self, x: Array) -> Array:
        diffs = x[:, None, :] - self.mu[None, :, :]  # (N, K, D)
        return -0.5 * jnp.sum(self.prec * diffs**2, axis=-1)  # (N, K)


class CrossAttention(Energy):
    Wk: Array  # (D, D)
    Wq: Array  # (D, D)

    def __init__(self, D: int, key: Array):
        k1, k2 = jr.split(key, 2)
        self.Wk = jr.normal(k1, (D, D))
        self.Wq = jr.normal(k2, (D, D))

    def measure(self, xq: Array, xk: Array) -> Array:
        Q = xq @ self.Wq  # (Nq, D)
        K = xk @ self.Wk  # (Nk, D)
        return Q @ K.T  # (Nq, Nk)


class Hopfield(Energy):
    def measure(self, x: Array, m: Array) -> Array:
        return x @ m.T


class SlotAttention(Energy):
    Wk: Array  # (D, D)
    Wq: Array  # (D, D)

    def __init__(self, D: int, key: Array):
        k1, k2 = jr.split(key, 2)
        self.Wk = jr.normal(k1, (D, D))
        self.Wq = jr.normal(k2, (D, D))

    def measure(self, x: Array, slots: Array) -> Array:
        K = x @ self.Wk  # (N, D)
        Q = slots @ self.Wq  # (S, D)
        return K @ Q.T  # (N, S)


class SelfAttention(Energy):
    Wk: Array  # (D, D)
    Wq: Array  # (D, D)

    def __init__(self, D: int, key: Array):
        k1, k2 = jr.split(key, 2)
        self.Wk = jr.normal(k1, (D, D))
        self.Wq = jr.normal(k2, (D, D))

    def measure(self, x: Array) -> Array:
        Q = x @ self.Wq
        K = x @ self.Wk
        return Q @ K.T
