"""Additional utility functions for ORC."""

import functools

import jax
import jax.numpy as jnp

@functools.partial(jax.jit, static_argnames=["max_iters"])
def max_eig_arnoldi_lax(A, tol=1e-12, max_iters=200, seed=0):
    '''
    Perform Arnoldi iteration to find the largest eigenvalue of a matrix A. Rough implementation rn 

    Args:
        A: The input matrix (n x n).
        tol: Tolerance for convergence.
        max_iters: Maximum number of iterations.
        seed: Random seed for initialization.
    Returns:
        The largest eigenvalue of A.
    '''
    n = A.shape[0]
    key = jax.random.PRNGKey(seed)
    v0 = jax.random.normal(key, (n,))
    v0 = v0 / jnp.linalg.norm(v0) # initial guess

    V = jnp.zeros((n, max_iters + 1)) # orthogonal krylov basis
    H = jnp.zeros((max_iters + 1, max_iters)) # hessenberg matrix
    V = V.at[:, 0].set(v0)

    def body(carry):
        j, V, H, prev_lam, done = carry
        w = A @ V[:, j]

        # Gram–Schmidt
        def gs_step(i, val):
            w, H = val
            h = jnp.vdot(V[:, i], w)
            H = H.at[i, j].set(h)
            w = w - h * V[:, i]
            return (w, H)
        w, H = jax.lax.fori_loop(0, j + 1, gs_step, (w, H))

        h_next = jnp.linalg.norm(w)
        H = H.at[j + 1, j].set(h_next)
        V = V.at[:, j + 1].set(jnp.where(h_next > 0, w / h_next, V[:, j + 1]))
        eigvals = jnp.linalg.eigvals(H[:-1, :])     # now square & static
        lam_max = eigvals[jnp.argmax(jnp.abs(eigvals))]

        done = jnp.logical_or(done, jnp.abs(lam_max - prev_lam) < tol) # not actually used
        return (j + 1, V, H, lam_max, done)

    # check against max_iters
    def cond(carry):
        j, _, _, _, done = carry
        keep_looping = jnp.logical_and(j < max_iters, jnp.logical_not(done))
        return keep_looping

    init_state = (0, V, H, 0.0 + 0j, False)
    j_final, _, _, lambda_max, _ = jax.lax.while_loop(cond, body, init_state)

    jax.lax.cond(
        j_final == max_iters,
        lambda: jax.debug.print(f"Arnoldi iteration did not converge in {max_iters} iterations. Consider increasing max_iters."),
        lambda: None,
    )
    return lambda_max
