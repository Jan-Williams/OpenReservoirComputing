import jax
import jax.numpy as jnp
import pytest

from orc.utils.regressions import (
    _solve_all_ridge_reg,
    _solve_all_ridge_reg_batched,
    ridge_regression,
)

ATOL = 1e-5

@pytest.fixture
def single_regression_data():
    """Generate test data for single ridge regression."""
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)

    seq_len, res_dim, out_dim = 100, 50, 3

    res_seq = jax.random.normal(key1, (seq_len, res_dim))
    true_cmat = jax.random.normal(key2, (out_dim, res_dim))
    noise = jax.random.normal(key3, (seq_len, out_dim)) * 0.01
    target_seq = res_seq @ true_cmat.T + noise

    return res_seq, target_seq, true_cmat


@pytest.fixture
def parallel_regression_data():
    """Generate test data for parallel ridge regression."""
    key = jax.random.PRNGKey(123)
    key1, key2, key3 = jax.random.split(key, 3)

    seq_len, chunks, res_dim, out_dim = 80, 10, 40, 2

    res_seq = jax.random.normal(key1, (seq_len, chunks, res_dim))
    true_cmat = jax.random.normal(key2, (chunks, out_dim, res_dim))
    noise = jax.random.normal(key3, (seq_len, chunks, out_dim)) * 0.01

    target_seq = jnp.einsum("tcr,cor->tco", res_seq, true_cmat) + noise

    return res_seq, target_seq, true_cmat


def test_ridge_regression_basic(single_regression_data):
    """Test basic ridge regression functionality."""
    res_seq, target_seq, _ = single_regression_data
    beta = 1e-6

    cmat = ridge_regression(res_seq, target_seq, beta)

    expected_shape = (target_seq.shape[1], res_seq.shape[1])
    assert cmat.shape == expected_shape

    reconstructed = res_seq @ cmat.T
    reconstruction_error = jnp.mean((reconstructed - target_seq) ** 2)
    assert reconstruction_error < 1e-3


def test_ridge_regression_regularization_effects(single_regression_data):
    """Test that higher regularization reduces overfitting."""
    res_seq, target_seq, _ = single_regression_data

    cmat_low = ridge_regression(res_seq, target_seq, beta=1e-8)
    recon_low = res_seq @ cmat_low.T
    error_low = jnp.mean((recon_low - target_seq) ** 2)

    cmat_high = ridge_regression(res_seq, target_seq, beta=1e-2)
    recon_high = res_seq @ cmat_high.T
    error_high = jnp.mean((recon_high - target_seq) ** 2)

    assert error_high > error_low

    assert jnp.linalg.norm(cmat_high) < jnp.linalg.norm(cmat_low)


def test_ridge_regression_zero_beta(single_regression_data):
    """Test ridge regression with zero regularization."""
    res_seq, target_seq, _ = single_regression_data

    U, s, Vt = jnp.linalg.svd(res_seq, full_matrices=False)
    min_singular_value = jnp.min(s)

    if min_singular_value > 1e-10:
        cmat = ridge_regression(res_seq, target_seq, beta=0.0)

        cmat_lstsq = jnp.linalg.lstsq(res_seq, target_seq, rcond=None)[0].T
        assert jnp.allclose(cmat, cmat_lstsq, atol=ATOL)


def test_ridge_regression_dtypes(single_regression_data):
    """Test that output dtype matches input dtype."""
    res_seq, target_seq, _ = single_regression_data

    res_seq_f32 = res_seq.astype(jnp.float32)
    target_seq_f32 = target_seq.astype(jnp.float32)
    cmat_f32 = ridge_regression(res_seq_f32, target_seq_f32)
    assert cmat_f32.dtype == jnp.float32

    res_seq_f64 = res_seq.astype(jnp.float64)
    target_seq_f64 = target_seq.astype(jnp.float64)
    cmat_f64 = ridge_regression(res_seq_f64, target_seq_f64)
    assert cmat_f64.dtype == jnp.float64


def test_solve_all_ridge_reg_basic(parallel_regression_data):
    """Test basic parallel ridge regression functionality."""
    res_seq, target_seq, _ = parallel_regression_data
    beta = 1e-6

    cmat_all = _solve_all_ridge_reg(res_seq, target_seq, beta)

    chunks, out_dim, res_dim = res_seq.shape[1], target_seq.shape[2], res_seq.shape[2]
    expected_shape = (chunks, out_dim, res_dim)
    assert cmat_all.shape == expected_shape


    for i in range(chunks):
        cmat_single = ridge_regression(res_seq[:, i, :], target_seq[:, i, :], beta)
        assert jnp.allclose(cmat_all[i], cmat_single, atol=ATOL)


def test_solve_all_ridge_reg_accuracy(parallel_regression_data):
    """Test reconstruction accuracy of parallel ridge regression."""
    res_seq, target_seq, _ = parallel_regression_data
    beta = 1e-6

    cmat_all = _solve_all_ridge_reg(res_seq, target_seq, beta)


    for i in range(res_seq.shape[1]):
        reconstructed = res_seq[:, i, :] @ cmat_all[i].T
        reconstruction_error = jnp.mean((reconstructed - target_seq[:, i, :]) ** 2)
        assert reconstruction_error < 1e-3


def test_batched_vs_unbatched_consistency(parallel_regression_data):
    """Test that batched and unbatched versions give identical results."""
    res_seq, target_seq, _ = parallel_regression_data
    beta = 1e-5
    chunks = res_seq.shape[1]

    cmat_unbatched = _solve_all_ridge_reg(res_seq, target_seq, beta)


    batch_sizes = [1, 3, 5, chunks // 2, chunks]

    for batch_size in batch_sizes:
        cmat_batched = _solve_all_ridge_reg_batched(
            res_seq, target_seq, beta, batch_size
        )

        assert jnp.allclose(cmat_batched, cmat_unbatched, atol=ATOL), (
            f"Batched (batch_size={batch_size}) and unbatched results differ"
        )


def test_batched_large_batch_size(parallel_regression_data):
    """Test that batch_size >= chunks delegates to unbatched version."""
    res_seq, target_seq, _ = parallel_regression_data
    beta = 1e-5
    chunks = res_seq.shape[1]

    cmat_unbatched = _solve_all_ridge_reg(res_seq, target_seq, beta)


    for batch_size in [chunks, chunks + 5, chunks * 2]:
        cmat_batched = _solve_all_ridge_reg_batched(
            res_seq, target_seq, beta, batch_size
        )

        assert jnp.allclose(cmat_batched, cmat_unbatched, atol=ATOL)


def test_batched_single_batch(parallel_regression_data):
    """Test batched version with batch_size=1."""
    res_seq, target_seq, _ = parallel_regression_data
    beta = 1e-5

    cmat_batched = _solve_all_ridge_reg_batched(res_seq, target_seq, beta, batch_size=1)


    for i in range(res_seq.shape[1]):
        cmat_single = ridge_regression(res_seq[:, i, :], target_seq[:, i, :], beta)
        assert jnp.allclose(cmat_batched[i], cmat_single, atol=ATOL)


def test_batched_prime_batch_size(parallel_regression_data):
    """Test batched version with batch size that doesn't divide evenly."""
    res_seq, target_seq, _ = parallel_regression_data
    beta = 1e-5

    batch_size = 3

    cmat_unbatched = _solve_all_ridge_reg(res_seq, target_seq, beta)
    cmat_batched = _solve_all_ridge_reg_batched(res_seq, target_seq, beta, batch_size)

    assert jnp.allclose(cmat_batched, cmat_unbatched, atol=ATOL)


def test_single_sample_regression():
    """Test ridge regression with single time step."""
    key = jax.random.PRNGKey(0)
    res_seq = jax.random.normal(key, (1, 10))
    target_seq = jax.random.normal(key, (1, 5))

    cmat = ridge_regression(res_seq, target_seq, beta=1e-3)
    assert cmat.shape == (5, 10)


def test_single_reservoir_dimension():
    """Test with single reservoir dimension."""
    key = jax.random.PRNGKey(0)
    res_seq = jax.random.normal(key, (50, 1))
    target_seq = jax.random.normal(key, (50, 3))

    cmat = ridge_regression(res_seq, target_seq, beta=1e-3)
    assert cmat.shape == (3, 1)


def test_single_output_dimension():
    """Test with single output dimension."""
    key = jax.random.PRNGKey(0)
    res_seq = jax.random.normal(key, (50, 10))
    target_seq = jax.random.normal(key, (50, 1))

    cmat = ridge_regression(res_seq, target_seq, beta=1e-3)
    assert cmat.shape == (1, 10)


def test_identical_targets():
    """Test regression when all targets are identical."""
    key = jax.random.PRNGKey(0)
    res_seq = jax.random.normal(key, (50, 10))
    target_seq = jnp.ones((50, 3))

    cmat = ridge_regression(res_seq, target_seq, beta=1e-3)
    assert cmat.shape == (3, 10)
    assert not jnp.any(jnp.isnan(cmat))
    assert not jnp.any(jnp.isinf(cmat))


def test_high_regularization(single_regression_data):
    """Test with very high regularization parameter."""
    res_seq, target_seq, _ = single_regression_data
    beta = 1e10

    cmat = ridge_regression(res_seq, target_seq, beta)


    assert jnp.linalg.norm(cmat) < 1e-5
    assert not jnp.any(jnp.isnan(cmat))
    assert not jnp.any(jnp.isinf(cmat))


def test_ill_conditioned_matrix():
    """Test with an ill-conditioned reservoir matrix."""
    key = jax.random.PRNGKey(0)

    base_col = jax.random.normal(key, (100, 1))
    noise = jax.random.normal(key, (100, 9)) * 1e-10
    res_seq = jnp.concatenate([base_col, base_col + noise], axis=1)

    target_seq = jax.random.normal(key, (100, 3))


    cmat = ridge_regression(res_seq, target_seq, beta=1e-3)
    assert not jnp.any(jnp.isnan(cmat))
    assert not jnp.any(jnp.isinf(cmat))


def test_different_beta_values(single_regression_data):
    """Test with various beta values to ensure numerical stability."""
    res_seq, target_seq, _ = single_regression_data

    beta_values = [0.0, 1e-12, 1e-8, 1e-4, 1e-2, 1.0, 1e2]

    for beta in beta_values:
        cmat = ridge_regression(res_seq, target_seq, beta)
        assert not jnp.any(jnp.isnan(cmat)), f"NaN values with beta={beta}"
        assert not jnp.any(jnp.isinf(cmat)), f"Inf values with beta={beta}"
