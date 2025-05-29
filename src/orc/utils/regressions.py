"""Implements common regressions used to train RC models."""

import jax
import jax.numpy as jnp
from jaxtyping import Array


def ridge_regression(res_seq: Array, target_seq: Array, beta: float):
    """Solve a single matrix ridge regression problem.

    Parameters
    ----------
    res_seq : Array
        Sequence of training reservoir states, (shape=(seq_len, res_dim)).
    target_seq : Array
        Sequence of training targe states, (shape=(seq_len, out_dim)).
    beta : float
        Tikhonov regularization parameter.

    Returns
    -------
    Array
        Solution to ridge regression s.t. cmat @ res_seq = target_seq.
    """
    lhs = res_seq.T @ res_seq + beta * jnp.eye(res_seq.shape[1], dtype=res_seq.dtype)
    rhs = res_seq.T @ target_seq
    cmat = jax.scipy.linalg.solve(lhs, rhs, assume_a="sym").T
    return cmat
