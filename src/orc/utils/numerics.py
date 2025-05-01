"""Additional utility functions for ORC."""

import functools

import jax
import jax.numpy as jnp
from jaxtyping import Array


@functools.partial(jax.jit, static_argnames=["max_iters"])
def _arnoldi_iteration(A,
                       max_iters=200,
                       seed=0):
    """Perform Arnoldi iteration to find an orthonormal basis for the Krylov subspace.

    Parameters
    ----------
    A : Array
        The input matrix (m x m) for which the Krylov subspace is computed.
    max_iters : int, optional
        The maximum number of Arnoldi iterations to perform. Default is 200. If the
        number of iterations exceeds the size of the matrix, it will be capped at m.
    seed : int, optional
        Random seed for initializing the starting vector. Default is 0.

    Returns
    -------
    Q : Array
        An orthonormal basis for the Krylov subspace (m x n).
    H : Array
        The upper Hessenberg matrix (n x n) representing the coefficients of the Krylov
        subspace.
    """
    # A is m x m; n is the size of the krylov basis
    m = A.shape[0]
    n = max_iters

    # choose a random vector to start iterating on
    key = jax.random.PRNGKey(seed)
    b = jax.random.normal(key, (m,))
    q0 = b / jnp.linalg.norm(b)

    # init krylov basis Q and hessenberg matrix H
    Q = jnp.zeros((m, n+1), dtype=A.dtype)
    H = jnp.zeros((n+1, n), dtype=A.dtype)
    Q = Q.at[:,0].set(q0)

    # modified gs step to form orth krylov basis
    def gs_step(carry, j):
        v, Q, mask = carry
        # Only apply when j is less than or equal to k (mask is True)
        h_jk = jnp.where(mask[j], jnp.dot(Q[:,j], v), 0.0)
        v = jnp.where(mask[j], v - h_jk*Q[:,j], v)
        return (v, Q, mask), h_jk

    def arnoldi_step(carry, k):
        A, Q, H = carry

        # new candidate vector
        v = A @ Q[:, k]

        # Create a mask for valid indices (0 to k)
        idx_mask = jnp.arange(n+1) <= k

        # run modified gs with fixed-size loop and masking
        final_carry_gs, h_jk_vals = jax.lax.scan(
            gs_step,
            (v, Q, idx_mask),
            jnp.arange(n+1)  # fixed size scan
        )
        v = final_carry_gs[0]  # orthogonalized candidate vector

        # Calculate subdiagonal
        h_kplus1k = jnp.linalg.norm(v)

        # Update H column k using a mask
        col_indices = jnp.arange(n+1)
        mask = col_indices <= k  # For the first k+1 elements

        # Apply h_jk_vals for the first k+1 elements using the mask
        H_col = jnp.where(mask, h_jk_vals, H[:, k])

        # Set the k+1 element to h_kplus1k
        H_col = H_col.at[k+1].set(h_kplus1k)

        # Update the k-th column of H
        H = H.at[:, k].set(H_col)

        # Update Q[:, k+1]
        Q = Q.at[:,k+1].set(v / (h_kplus1k))

        return (A, Q, H), None

    final_carry_arnoldi, _ = jax.lax.scan(arnoldi_step, (A, Q, H), jnp.arange(n))
    _, Q, H = final_carry_arnoldi

    return Q[:,:n], H[:n,:n]

@functools.partial(jax.jit, static_argnames=("max_iters",))
def max_eig_arnoldi(A: Array,
                    max_iters: int = 200,
                    seed: int = 0):
    """Compute the maximum eigenvalue of a matrix using the Arnoldi iteration method.

    Parameters
    ----------
    A : Array
        The input matrix (m x m) for which the maximum eigenvalue is computed.
    max_iters : int, optional
        The maximum number of Arnoldi iterations to perform. Default is 200. If the
        number of iterations exceeds the size of the matrix, it will be capped at m.
    seed : int, optional
        Random seed for initializing the starting vector. Default is 0.
    """
    _,H = _arnoldi_iteration(A, max_iters, seed)
    eigvals = jnp.linalg.eigvals(H)
    lambda_max = eigvals[jnp.argmax(jnp.abs(eigvals))]
    return lambda_max
