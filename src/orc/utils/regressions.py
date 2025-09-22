"""Implements common regressions used to train RC models."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array


def ridge_regression(res_seq: Array, target_seq: Array, beta: float = 1e-7):
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


_solve_all_ridge_reg = eqx.filter_vmap(ridge_regression, in_axes=eqx.if_array(1))


def _solve_all_ridge_reg_batched(
    res_seq_train: Array, target_seq: Array, beta: float, batch_size: int
) -> Array:
    """Solve ridge regression for all parallel reservoirs using batched vmap.

    This function processes the parallel reservoirs in batches to reduce memory
    usage for large numbers of parallel reservoirs.

    Parameters
    ----------
    res_seq_train : Array
        Training reservoir states, shape=(seq_len, chunks, res_dim).
    target_seq : Array
        Target sequence, shape=(seq_len, chunks, out_dim).
    beta : float
        Tikhonov regularization parameter.
    batch_size : int
        Number of parallel reservoirs to process in each batch.

    Returns
    -------
    Array
        Ridge regression solution for all chunks, shape=(chunks, out_dim, res_dim).
    """
    chunks = res_seq_train.shape[1]

    if batch_size >= chunks:
        return _solve_all_ridge_reg(res_seq_train, target_seq, beta)

    results = []
    for i in range(0, chunks, batch_size):
        end_idx = min(i + batch_size, chunks)
        batch_res = res_seq_train[:, i:end_idx, :]
        batch_target = target_seq[:, i:end_idx, :]

        batch_result = _solve_all_ridge_reg(batch_res, batch_target, beta)
        results.append(batch_result)

    return jnp.concatenate(results, axis=0)
