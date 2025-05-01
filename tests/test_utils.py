import jax
import jax.experimental.sparse
import jax.numpy as jnp
import pytest

from orc.utils import max_eig_arnoldi

# Tolerance for numerical comparisons
ATOL = 1e-5

@pytest.fixture(params=
                [(50, 0.1), (100, 0.02), (1000, 0.01), (2000, 0.01)]
                )
def matrix_parms(request):
    return request.param

def random_sparse_matrix(n, density):
    key = jax.random.PRNGKey(999)
    A = jax.experimental.sparse.random_bcoo(key=key,
                                            shape=(n, n),
                                            nse=density,
                                            dtype=jnp.float64,
                                            generator = jax.random.normal)
    return A

def test_max_eig_diag(matrix_parms):
    """Test max_eig_arnoldi with a diagonal matrix."""
    n, _ = matrix_parms
    key = jax.random.PRNGKey(0)
    eigs = jax.random.normal(key, (n,))
    A = jnp.diag(eigs)
    max_eig = max_eig_arnoldi(A)
    assert jnp.isclose(jnp.abs(max_eig), jnp.max(jnp.abs(eigs)), atol=ATOL)

def test_max_eig_random_sparse(matrix_parms):
    """Test max_eig_arnoldi with a random sparse matrix."""
    n, density = matrix_parms
    A = random_sparse_matrix(n, density)
    max_eig = max_eig_arnoldi(A)
    eigvals = jnp.linalg.eigvals(A.todense())
    assert jnp.isclose(jnp.abs(max_eig), jnp.max(jnp.abs(eigvals)), atol=ATOL)

def test_max_eig_seed_consistency(matrix_parms):
    """Test that the same seed produces the same max eigenvalue approximation."""
    n, density = matrix_parms
    A = random_sparse_matrix(n, density)

    eig1 = max_eig_arnoldi(A, seed=42)
    eig2 = max_eig_arnoldi(A, seed=42)

    assert jnp.isclose(eig1, eig2) # no tolerance
