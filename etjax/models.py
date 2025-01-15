import equinox as eqx
from jax import Array, numpy as jnp, random as jr, nn


class SelfAttention(eqx.Module):
    Wq: Array
    Wk: Array
    query_dim: int = eqx.field(static=True)

    def __init__(self, num_heads: int, dim: int, query_dim: int, key: jr.PRNGKey):
        super().__init__()
        k1, k2 = jr.split(key)
        self.Wk = jr.normal(k1, (dim, num_heads, query_dim))
        self.Wq = jr.normal(k2, (dim, num_heads, query_dim))
        self.query_dim = query_dim

    def __call__(self, x: Array) -> Array:
        beta = 1 / jnp.sqrt(self.query_dim)
        K = jnp.einsum("kd,hzd->khz", x, self.Wk)
        Q = jnp.einsum("qd,hzd->qhz", x, self.Wq)
        A = nn.logsumexp(beta * jnp.einsum("qhz,khz->hqk", Q, K), -1)
        return -1.0 / beta * jnp.sum(A)


class SlotAttention(eqx.Module):
    Wk: Array
    Wq: Array

    def __init__(self, dim: int, key: jr.PRNGKey):
        super().__init__()
        k1, k2 = jr.split(key, 2)
        self.Wk = jr.normal(k1, (dim, dum))
        self.Wq = jr.normal(k2, (D, D))

    def __call__(self, x: Array, z: Array) -> Array:
        N, D = x.shape
        S = z.shape[0]
        beta = 1.0 / jnp.sqrt(D)

        K = x @ self.Wk
        Q = z @ self.Wq
        A = jnp.einsum("nd,sd->ns", K, Q)
        A = nn.logsumexp(beta * A, axis=1)
        return -1.0 / beta * jnp.sum(A)


class Hopfield(eqx.Module):
    Xi: jax.Array

    def __init__(self, dim: int, num_mems: int, key: jr.PRNGKey):
        self.Xi = jr.normal(key, (dim, num_mems))

    def __call__(self, x: Array):
        hid = jnp.einsum("nd,dm->nm", g, self.Xi)
        return -0.5 * (nn.relu(hid) ** 2).sum()
