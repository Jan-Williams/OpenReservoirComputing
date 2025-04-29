"""Additional utility functions for ORC."""

import functools

import jax
import jax.numpy as jnp
from jaxtyping import Array


@functools.partial(jax.jit, static_argnames=("max_iters",))
def _arnoldi_iteration(A: Array, 
                       max_iters: int = 200, 
                       seed: int = 0):
    """ Performs the Arnoldi iteration to compute an orthonormal basis for the Krylov subspace.

    Parameters
    ----------
    A : Array
        The input matrix (m x m) for which the Krylov subspace is computed.
    max_iters : int, optional
        The maximum number of Arnoldi iterations to perform. Default is 200. If the number of iterations exceeds the size of the matrix, it will be capped at m.
    seed : int, optional
        Random seed for initializing the starting vector. Default is 0.
    
    Returns
    -------
    Q : Array
        An orthonormal basis for the Krylov subspace (m x n).
    H : Array
        The upper Hessenberg matrix (n x n) representing the coefficients of the Krylov subspace.
    """
    # A is m x m; n is the size of the krylov basis
    m = A.shape[0]
    n = min(max_iters, m)

    # choose a random vector to start iterating on 
    key = jax.random.PRNGKey(seed)
    q0  = jax.random.normal(key, (m,))
    q0  = q0 / jnp.linalg.norm(q0)

    # init krylov basis Q and hessenberg matrix H
    Q = jnp.zeros((m, n + 1), dtype=A.dtype)
    H = jnp.zeros((n + 1, n), dtype=A.dtype)
    Q = Q.at[:, 0].set(q0)

    # run arnoldi one arnoldi step for an entire column of H
    col_idx = jnp.arange(n + 1)
    def arnoldi_col_step(carry, k):
        Q, H = carry

        # new candidate vector
        v = A @ Q[:, k]                       

        # orthogonalize in a batch 
        h_full = jnp.dot(Q.T, v) # all inner products
        h_mask = (col_idx <= k)                 
        h = jnp.where(h_mask, h_full, 0)  # zeros beyond k
        v = v - Q @ h
        beta = jnp.linalg.norm(v)

        # build the whole column 
        h_col= h.at[k+1].set(beta) # subdiag is normed   
        H  = H.at[:, k].set(h_col)           

        Q  = Q.at[:, k+1].set(v / beta) 
        return (Q, H), None

    (Q, H), _ = jax.lax.scan(arnoldi_col_step, (Q, H), jnp.arange(n))
    return Q[:, :n], H[:n, :n]

@functools.partial(jax.jit, static_argnames=("max_iters",))
def max_eig_arnoldi(A: Array, 
                    max_iters: int = 200, 
                    seed: int = 0):
    """ Computes the maximum eigenvalue of a matrix using the Arnoldi iteration method.

    Parameters
    ----------
    A : Array
        The input matrix (m x m) for which the maximum eigenvalue is computed.
    max_iters : int, optional
        The maximum number of Arnoldi iterations to perform. Default is 200. If the number of iterations exceeds the size of the matrix, it will be capped at m.
    seed : int, optional
        Random seed for initializing the starting vector. Default is 0.
    """
    _,H = _arnoldi_iteration(A, max_iters, seed)
    eigvals = jnp.linalg.eigvals(H)
    lambda_max = eigvals[jnp.argmax(jnp.abs(eigvals))]
    return lambda_max
