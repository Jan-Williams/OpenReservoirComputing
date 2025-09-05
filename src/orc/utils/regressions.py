"""Implements common regressions used to train RC models."""

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
from jaxtyping import Array


def ridge_regression_direct(res_seq: Array, target_seq: Array, beta: float=1e-7):
    """Solve a single matrix ridge regression problem.

    Parameters
    ----------
    res_seq : Array
        Sequence of training reservoir states, (shape=(seq_len, res_dim)).
    target_seq : Array
        Sequence of training target states, (shape=(seq_len, out_dim)).
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

def ridge_regression_cg(res_seq: Array, target_seq: Array, beta: float = 1e-7,
                        tol: float = 1e-6, maxiter: int | None = None):
    """Solve a ridge regression problem iteratively using Conjugate Gradient (CG).

    This function uses CG to avoid explicit matrix factorization.
    Use when reservoir dimension is large or sparse.

    Parameters
    ----------
    res_seq : Array
        Sequence of training reservoir states, (shape=(seq_len, res_dim)).
    target_seq : Array
        Sequence of training target states, (shape=(seq_len, out_dim)).
    beta : float
        Tikhonov regularization parameter.
    tol : float
        Convergence tolerance for the CG solver.
    maxiter : int or None
        Maximum number of iterations, or None for no limit.

    Returns
    -------
    Array
        Solution to ridge regression s.t. cmat @ res_seq = target_seq.
    """
    def matvec(v):
        return res_seq.T @ (res_seq @ v) + beta * v

    def solve_one(rhs_col):
        x0 = jnp.zeros_like(rhs_col)
        sol, _ = cg(matvec, rhs_col, x0=x0, tol=tol, maxiter=maxiter)
        return sol

    rhs = res_seq.T @ target_seq  # shape=(res_dim, out_dim)
    w = jax.vmap(solve_one, in_axes=1, out_axes=1)(rhs)  # [res_dim, out_dim]
    return w.T

def ridge_regression_cholesky(res_seq: Array, target_seq: Array, beta: float = 1e-7):
    """Solve a ridge regression problem directly using Cholesky factorization.

    This function uses Cholesky on the SPD normal matrix.
    Use when reservoir dimension is moderate and dense.

    Parameters
    ----------
    res_seq : Array
        Sequence of training reservoir states, (shape=(seq_len, res_dim)).
    target_seq : Array
        Sequence of training target states, (shape=(seq_len, out_dim)).
    beta : float
        Tikhonov regularization parameter.

    Returns
    -------
    Array
        Solution to ridge regression s.t. cmat @ res_seq = target_seq.
    """
    lhs = res_seq.T @ res_seq + beta * jnp.eye(res_seq.shape[1], dtype=res_seq.dtype)
    rhs = res_seq.T @ target_seq
    chol = jnp.linalg.cholesky(lhs)
    sol = jax.scipy.linalg.cho_solve((chol, True), rhs)
    return sol.T

def ridge_regression(res_seq: Array, target_seq: Array, beta: float = 1e-7,
                method: str = "direct", tol: float = 1e-6, maxiter: int | None = None):
    """Solve a ridge regression problem using different solution methods.

    Parameters
    ----------
    res_seq : Array
        Sequence of training reservoir states, (shape=(seq_len, res_dim)).
    target_seq : Array
        Sequence of training target states, (shape=(seq_len, out_dim)).
    beta : float
        Tikhonov regularization parameter.
    method : {"direct", "cg", "cholesky"}
        Solver method:
        - "direct"   : general linear solver for symmetric matrices, not exploiting SPD.
        - "cg"       : iterative Conjugate Gradient, avoids explicit factorization.
        - "cholesky" : direct Cholesky factorization for SPD systems.
    tol : float
        Convergence tolerance (used only for CG).
    maxiter : int or None
        Maximum iterations for CG, or None for no limit.

    Returns
    -------
    Array
        Solution to ridge regression s.t. cmat @ res_seq = target_seq.
    """
    if method == "direct":
        return ridge_regression_direct(res_seq, target_seq, beta)
    elif method == "cg":
        return ridge_regression_cg(res_seq, target_seq, beta, tol=tol, maxiter=maxiter)
    elif method == "cholesky":
        return ridge_regression_cholesky(res_seq, target_seq, beta)
    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Choose from 'direct', 'cg', 'cholesky'.")
