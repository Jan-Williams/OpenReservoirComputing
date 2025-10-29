import jax
import jax.numpy as jnp
import pytest
from jax.experimental import sparse

import orc

##################### ESN TESTS #####################


@pytest.fixture
def esndriver():
    return orc.drivers.ParallelESNDriver(
        res_dim=212,
        leak=0.123,
        spectral_radius=0.6,
        density=0.2,
        bias=1.6,
        dtype=jnp.float64,
        seed=0,
    )


def test_esndriver_dims(esndriver):
    key = jax.random.key(999)
    res_dim = esndriver.res_dim
    test_vec = jax.random.normal(key, shape=(1, res_dim))
    out_vec = esndriver.advance(test_vec, test_vec)
    assert out_vec.shape == (
        1,
        res_dim,
    )

    test_vec = jax.random.normal(key, shape=(1, res_dim - 1))
    with pytest.raises(ValueError):
        out_vec = esndriver.advance(test_vec, test_vec)


@pytest.mark.parametrize("batch_size", [3, 12, 52])
def test_batchapply_dims_esn(batch_size, esndriver):
    key = jax.random.key(42)
    res_dim = esndriver.res_dim
    test_vec = jax.random.normal(key, shape=(batch_size, 1, res_dim))
    out_vec = esndriver.batch_advance(test_vec, test_vec)

    assert out_vec.shape == (batch_size, 1, res_dim)

    test_vec = jax.random.normal(key, shape=(batch_size, 1, res_dim - 1))

    with pytest.raises(ValueError):
        out_vec = esndriver.batch_advance(test_vec, test_vec)


@pytest.mark.parametrize(
    "res_dim,leak,spectral_radius,density,bias,dtype",
    [
        (22, 0.123, 0.6, 0.02, 1.6, jnp.int32),
        (22.2, 0.123, 0.6, 0.02, 1.6, jnp.float64),
    ],
)
def test_param_types_esn(res_dim, leak, spectral_radius, density, bias, dtype):
    with pytest.raises(TypeError):
        _ = orc.drivers.ParallelESNDriver(
            res_dim=res_dim,
            leak=leak,
            spectral_radius=spectral_radius,
            density=density,
            bias=bias,
            dtype=dtype,
            seed=33,
        )


@pytest.mark.parametrize(
    "res_dim,leak,spectral_radius,density,bias,dtype",
    [
        (22, 0.123, -0.5, 0.02, 1.6, jnp.float32),
        (22, 0.123, 0.6, 1.3, 1.6, jnp.float64),
        (22, -0.2, 0.6, 0.04, 1.6, jnp.float32),
    ],
)
def test_param_vals_esn(res_dim, leak, spectral_radius, density, bias, dtype):
    with pytest.raises(ValueError):
        _ = orc.drivers.ParallelESNDriver(
            res_dim=res_dim,
            leak=leak,
            spectral_radius=spectral_radius,
            density=density,
            bias=bias,
            dtype=dtype,
            seed=32,
        )


@pytest.mark.parametrize("chunks", [2, 4, 8, 9])
def test_call_ones_esn(chunks):
    model = orc.drivers.ParallelESNDriver(
        res_dim=212,
        leak=0.123,
        spectral_radius=0.6,
        density=0.2,
        bias=1.6,
        dtype=jnp.float64,
        seed=0,
        chunks=chunks,
    )
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key)
    test_vec1 = jax.random.normal(key=key1, shape=(chunks, 212))
    test_vec2 = jax.random.normal(key=key2, shape=(chunks, 212))
    test_outputs = model(test_vec1, test_vec2)
    wr = sparse.BCOO.todense(model.wr)
    bias = model.bias
    leak = model.leak

    def naive_imp_forward(wr, bias, leak, proj_vars, res_state):
        res_next = jnp.tanh(
            wr @ res_state + proj_vars + bias * jnp.ones_like(proj_vars)
        )
        return leak * res_next + (1 - leak) * res_state

    gt_outputs = jnp.empty((chunks, 212))
    for group in range(chunks):
        gt = naive_imp_forward(
            wr[group], bias, leak, test_vec1[group], test_vec2[group]
        )
        gt_outputs = gt_outputs.at[group].set(gt)
    assert jnp.allclose(gt_outputs, test_outputs)


@pytest.mark.parametrize(
    "res_dim, spectral_radius, density, chunks",
    [
        (100, 0.876, 0.02, 1),
        (500, 0.546, 0.01, 1),
        (1000, 0.432, 0.01, 1),
        (1000, 0.1, 0.01, 1),
        (1000, 1.3, 0.01, 1),
        (100, 0.345, 0.02, 15),
        (500, 0.673, 0.01, 4),
    ],
)
def test_driver_spectral_radius_sparse(res_dim, spectral_radius, density, chunks):
    """Test that the spectral radius of the reservoir update matrix is as expected."""
    driver = orc.drivers.ParallelESNDriver(
        res_dim=res_dim,
        spectral_radius=spectral_radius,
        density=density,
        dtype=jnp.float64,
        seed=0,
        chunks=chunks,
        use_sparse_eigs=True,
    )

    wr = driver.wr
    wr_max_eigs = jnp.max(jnp.abs(jnp.linalg.eigvals(wr.todense())), axis=1)
    assert jnp.isclose(wr_max_eigs, spectral_radius, atol=1e-5).all(), (
        f"Expected spectral radius {spectral_radius}, but got {wr_max_eigs}"
    )


@pytest.mark.parametrize(
    "res_dim, spectral_radius, density, chunks",
    [
        (10, 0.6, 0.5, 1),
        (50, 0.6, 0.1, 1),
        (100, 0.876, 0.01, 1),
        (1000, 0.432, 0.01, 1),
        (1000, 0.1, 0.01, 1),
        (1000, 1.3, 0.01, 1),
        (100, 0.345, 0.02, 15),
        (500, 0.673, 0.01, 4),
    ],
)
def test_driver_spectral_radius_dense(res_dim, spectral_radius, density, chunks):
    """Test that the spectral radius of the reservoir update matrix is as expected"""
    driver = orc.drivers.ParallelESNDriver(
        res_dim=res_dim,
        spectral_radius=spectral_radius,
        density=density,
        dtype=jnp.float64,
        seed=0,
        chunks=chunks,
        use_sparse_eigs=False,
    )

    wr = driver.wr
    wr_max_eigs = jnp.max(jnp.abs(jnp.linalg.eigvals(wr.todense())), axis=1)
    assert jnp.isclose(wr_max_eigs, spectral_radius, atol=1e-5).all(), (
        f"Expected spectral radius {spectral_radius}, but got {wr_max_eigs}"
    )


##################### ESN BATCHED EIGENVALUE TESTS #####################


@pytest.mark.parametrize(
    "res_dim, spectral_radius, density, chunks, batch_size",
    [
        (100, 0.8, 0.1, 10, 3),
        (150, 0.6, 0.05, 20, 5),
        (200, 0.9, 0.02, 50, 10),
        (100, 0.7, 0.1, 100, 25),
    ],
)
def test_batched_eigenvals_sparse_equivalence(
    res_dim, spectral_radius, density, chunks, batch_size
):
    """Test that batched sparse computation gives same results as non-batched."""
    seed = 42

    # Create driver without batching (default behavior)
    driver_unbatched = orc.drivers.ParallelESNDriver(
        res_dim=res_dim,
        spectral_radius=spectral_radius,
        density=density,
        chunks=chunks,
        seed=seed,
        use_sparse_eigs=True,
    )

    # Create driver with batching
    driver_batched = orc.drivers.ParallelESNDriver(
        res_dim=res_dim,
        spectral_radius=spectral_radius,
        density=density,
        chunks=chunks,
        seed=seed,
        use_sparse_eigs=True,
        eigenval_batch_size=batch_size,
    )

    # Check that spectral radii are equivalent
    wr_unbatched = driver_unbatched.wr
    wr_batched = driver_batched.wr

    unbatched_eigs = jnp.max(
        jnp.abs(jnp.linalg.eigvals(wr_unbatched.todense())), axis=1
    )
    batched_eigs = jnp.max(jnp.abs(jnp.linalg.eigvals(wr_batched.todense())), axis=1)

    # Both should achieve the target spectral radius
    assert jnp.allclose(unbatched_eigs, spectral_radius, atol=1e-5)
    assert jnp.allclose(batched_eigs, spectral_radius, atol=1e-5)

    # Results should be identical (same seed, same computation)
    assert jnp.allclose(unbatched_eigs, batched_eigs, atol=1e-10)


@pytest.mark.parametrize(
    "res_dim, spectral_radius, density, chunks, batch_size",
    [
        (150, 0.8, 0.1, 10, 3),
        (100, 0.6, 0.05, 20, 7),
        (200, 0.9, 0.02, 30, 8),
    ],
)
def test_batched_eigenvals_dense_equivalence(
    res_dim, spectral_radius, density, chunks, batch_size
):
    """Test that batched dense computation gives same results as non-batched."""
    seed = 123

    # Create driver without batching (default behavior)
    driver_unbatched = orc.drivers.ParallelESNDriver(
        res_dim=res_dim,
        spectral_radius=spectral_radius,
        density=density,
        chunks=chunks,
        seed=seed,
        use_sparse_eigs=False,
    )

    # Create driver with batching
    driver_batched = orc.drivers.ParallelESNDriver(
        res_dim=res_dim,
        spectral_radius=spectral_radius,
        density=density,
        chunks=chunks,
        seed=seed,
        use_sparse_eigs=False,
        eigenval_batch_size=batch_size,
    )

    # Check that spectral radii are equivalent
    wr_unbatched = driver_unbatched.wr
    wr_batched = driver_batched.wr

    unbatched_eigs = jnp.max(
        jnp.abs(jnp.linalg.eigvals(wr_unbatched.todense())), axis=1
    )
    batched_eigs = jnp.max(jnp.abs(jnp.linalg.eigvals(wr_batched.todense())), axis=1)

    # Both should achieve the target spectral radius
    assert jnp.allclose(unbatched_eigs, spectral_radius, atol=1e-5)
    assert jnp.allclose(batched_eigs, spectral_radius, atol=1e-5)

    # Results should be identical (same seed, same computation)
    assert jnp.allclose(unbatched_eigs, batched_eigs, atol=1e-10)


def test_batched_eigenvals_large_batch_size():
    """Test that batch size larger than chunks works correctly."""
    res_dim, chunks = 150, 5
    large_batch_size = 20  # Larger than chunks

    driver = orc.drivers.ParallelESNDriver(
        res_dim=res_dim,
        spectral_radius=0.8,
        density=0.1,
        chunks=chunks,
        seed=999,
        eigenval_batch_size=large_batch_size,
    )

    # Should work without issues and achieve target spectral radius
    wr = driver.wr
    eigs = jnp.max(jnp.abs(jnp.linalg.eigvals(wr.todense())), axis=1)
    assert jnp.allclose(eigs, 0.8, atol=1e-5)


def test_batched_eigenvals_single_chunk():
    """Test batching with single chunk (edge case)."""
    driver = orc.drivers.ParallelESNDriver(
        res_dim=150,
        spectral_radius=0.7,
        density=0.05,
        chunks=1,
        seed=777,
        eigenval_batch_size=5,
    )

    # Should work correctly for single chunk
    wr = driver.wr
    eigs = jnp.max(jnp.abs(jnp.linalg.eigvals(wr.todense())), axis=1)
    assert jnp.allclose(eigs, 0.7, atol=1e-5)


def test_batched_eigenvals_batch_size_one():
    """Test batching with batch size of 1 (most memory efficient)."""
    chunks = 10
    driver = orc.drivers.ParallelESNDriver(
        res_dim=150,
        spectral_radius=0.9,
        density=0.1,
        chunks=chunks,
        seed=555,
        eigenval_batch_size=1,
    )

    # Should process each matrix individually and achieve target spectral radius
    wr = driver.wr
    eigs = jnp.max(jnp.abs(jnp.linalg.eigvals(wr.todense())), axis=1)
    assert jnp.allclose(eigs, 0.9, atol=1e-5)
    assert len(eigs) == chunks


def test_batched_eigenvals_functionality_preserved():
    """Test that batched drivers maintain full ESN functionality."""
    driver = orc.drivers.ParallelESNDriver(
        res_dim=100,
        spectral_radius=0.8,
        density=0.05,
        chunks=5,
        seed=333,
        eigenval_batch_size=2,
    )

    # Test basic advance functionality
    key = jax.random.key(42)
    proj_vars = jax.random.normal(key, shape=(5, 100))
    res_state = jax.random.normal(key, shape=(5, 100))

    # Should advance without issues
    new_state = driver.advance(proj_vars, res_state)
    assert new_state.shape == (5, 100)

    # Test batch advance
    batch_proj_vars = jax.random.normal(key, shape=(3, 5, 100))
    batch_res_state = jax.random.normal(key, shape=(3, 5, 100))

    batch_new_state = driver.batch_advance(batch_proj_vars, batch_res_state)
    assert batch_new_state.shape == (3, 5, 100)


##################### SINGLE ESN DRIVER TESTS #####################


@pytest.fixture
def single_esndriver():
    return orc.drivers.ESNDriver(
        res_dim=100,
        leak=0.3,
        spectral_radius=0.9,
        density=0.1,
        bias=1.0,
        dtype=jnp.float64,
        seed=42,
    )


def test_single_esndriver_dims(single_esndriver):
    """Test that ESNDriver works with single reservoir (no chunks dimension)."""
    key = jax.random.key(123)
    res_dim = single_esndriver.res_dim

    # Test single state advance
    proj_vars = jax.random.normal(key, shape=(res_dim,))
    res_state = jax.random.normal(key, shape=(res_dim,))
    out_state = single_esndriver.advance(proj_vars, res_state)

    assert out_state.shape == (res_dim,)
    assert jnp.all(jnp.isfinite(out_state))


def test_single_esndriver_call(single_esndriver):
    """Test ESNDriver __call__ method handles both single and batch inputs."""
    key = jax.random.key(456)
    res_dim = single_esndriver.res_dim

    # Test single input
    proj_vars = jax.random.normal(key, shape=(res_dim,))
    res_state = jax.random.normal(key, shape=(res_dim,))
    out_state = single_esndriver(proj_vars, res_state)
    assert out_state.shape == (res_dim,)

    # Test batch input
    batch_proj_vars = jax.random.normal(key, shape=(5, res_dim))
    batch_res_state = jax.random.normal(key, shape=(5, res_dim))
    batch_out = single_esndriver(batch_proj_vars, batch_res_state)
    assert batch_out.shape == (5, res_dim)


def test_single_esndriver_chunks_is_one(single_esndriver):
    """Test that ESNDriver always has chunks=1."""
    assert single_esndriver.chunks == 1


##################### CESN TESTS #####################


@pytest.fixture
def cesn_driver():
    return orc.drivers.ParallelESNDriver(
        res_dim=200,
        time_const=50.0,  # Continuous parameter
        spectral_radius=0.7,
        density=0.2,
        bias=1.6,
        dtype=jnp.float64,
        seed=0,
        mode="continuous",  # Set continuous mode
    )


def test_cesn_driver_dims(cesn_driver):
    key = jax.random.key(999)
    res_dim = cesn_driver.res_dim
    test_vec = jax.random.normal(key, shape=(1, res_dim))
    out_vec = cesn_driver.advance(test_vec, test_vec)
    assert out_vec.shape == (
        1,
        res_dim,
    )

    test_vec = jax.random.normal(key, shape=(1, res_dim - 1))
    with pytest.raises(ValueError):
        out_vec = cesn_driver.advance(test_vec, test_vec)


@pytest.mark.parametrize("batch_size", [3, 12, 52])
def test_batchapply_dims_cesn(batch_size, cesn_driver):
    key = jax.random.key(42)
    res_dim = cesn_driver.res_dim
    test_vec = jax.random.normal(key, shape=(batch_size, 1, res_dim))
    out_vec = cesn_driver.batch_advance(test_vec, test_vec)

    assert out_vec.shape == (batch_size, 1, res_dim)

    test_vec = jax.random.normal(key, shape=(batch_size, 1, res_dim - 1))
    with pytest.raises(ValueError):
        out_vec = cesn_driver.batch_advance(test_vec, test_vec)


@pytest.mark.parametrize(
    "res_dim,time_const,spectral_radius,density,bias,dtype",
    [
        (22, 50.0, 0.6, 0.02, 1.6, jnp.int32),
        (22.2, 50.0, 0.6, 0.02, 1.6, jnp.float64),
    ],
)
def test_param_types_cesn(res_dim, time_const, spectral_radius, density, bias, dtype):
    with pytest.raises(TypeError):
        _ = orc.drivers.ParallelESNDriver(
            res_dim=res_dim,
            time_const=time_const,
            spectral_radius=spectral_radius,
            density=density,
            bias=bias,
            dtype=dtype,
            seed=33,
            mode="continuous",
        )


@pytest.mark.parametrize(
    "res_dim,time_const,spectral_radius,density,bias,dtype",
    [
        (22, 50.0, -0.5, 0.02, 1.6, jnp.float32),
        (22, 50.0, 0.6, 1.3, 1.6, jnp.float64),
        (22, -1.0, 0.6, 0.04, 1.6, jnp.float32),
    ],
)
def test_param_vals_cesn(res_dim, time_const, spectral_radius, density, bias, dtype):
    with pytest.raises(ValueError):
        _ = orc.drivers.ParallelESNDriver(
            res_dim=res_dim,
            time_const=time_const,
            spectral_radius=spectral_radius,
            density=density,
            bias=bias,
            dtype=dtype,
            seed=32,
            mode="continuous",
        )


@pytest.mark.parametrize("chunks", [2, 4, 8])
def test_call_cesn(chunks):
    model = orc.drivers.ParallelESNDriver(
        res_dim=150,
        time_const=50.0,
        spectral_radius=0.6,
        density=0.2,
        bias=1.6,
        dtype=jnp.float64,
        seed=0,
        chunks=chunks,
        mode="continuous",
    )
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key)
    test_vec1 = jax.random.normal(key=key1, shape=(chunks, 150))
    test_vec2 = jax.random.normal(key=key2, shape=(chunks, 150))

    # Get the model output
    test_outputs = model(test_vec1, test_vec2)

    # Check shape of outputs
    assert test_outputs.shape == (chunks, 150)

    # Verify output values are finite
    assert jnp.all(jnp.isfinite(test_outputs))


##################### TAYLOR DRIVER TESTS #####################


@pytest.fixture
def taylordriver():
    return orc.drivers.ParallelTaylorDriver(
        n_terms=3,
        res_dim=212,
        spectral_radius=0.6,
        density=0.2,
        bias=1.6,
        dtype=jnp.float64,
        seed=0,
    )


def test_taylordriver_dims(taylordriver):
    key = jax.random.key(999)
    res_dim = taylordriver.res_dim
    test_vec = jax.random.normal(key, shape=(1, res_dim))
    out_vec = taylordriver.advance(test_vec, test_vec)
    assert out_vec.shape == (
        1,
        res_dim,
    )

    test_vec = jax.random.normal(key, shape=(1, res_dim - 1))
    with pytest.raises(ValueError):
        out_vec = taylordriver.advance(test_vec, test_vec)


@pytest.mark.parametrize("batch_size", [3, 12, 52])
def test_batchapply_dims_taylor(batch_size, taylordriver):
    key = jax.random.key(42)
    res_dim = taylordriver.res_dim
    test_vec = jax.random.normal(key, shape=(batch_size, 1, res_dim))
    out_vec = taylordriver.batch_advance(test_vec, test_vec)

    assert out_vec.shape == (batch_size, 1, res_dim)

    test_vec = jax.random.normal(key, shape=(batch_size, 1, res_dim - 1))

    with pytest.raises(ValueError):
        out_vec = taylordriver.batch_advance(test_vec, test_vec)


@pytest.mark.parametrize(
    "n_terms,res_dim,spectral_radius,density,bias,dtype",
    [
        (3, 22, 0.6, 0.02, 1.6, jnp.int32),
        (2, 22.2, 0.6, 0.02, 1.6, jnp.float64),
    ],
)
def test_param_types_taylor(n_terms, res_dim, spectral_radius, density, bias, dtype):
    with pytest.raises(TypeError):
        _ = orc.drivers.ParallelTaylorDriver(
            n_terms=n_terms,
            res_dim=res_dim,
            spectral_radius=spectral_radius,
            density=density,
            bias=bias,
            dtype=dtype,
            seed=33,
        )


@pytest.mark.parametrize(
    "n_terms,res_dim,spectral_radius,density,bias,dtype",
    [
        (3, 22, -0.5, 0.02, 1.6, jnp.float32),
        (2, 22, 0.6, 1.3, 1.6, jnp.float64),
        (6, 22, 0.6, 0.04, 1.6, jnp.float32),
    ],
)
def test_param_vals_taylor(n_terms, res_dim, spectral_radius, density, bias, dtype):
    with pytest.raises(ValueError):
        _ = orc.drivers.ParallelTaylorDriver(
            n_terms=n_terms,
            res_dim=res_dim,
            spectral_radius=spectral_radius,
            density=density,
            bias=bias,
            dtype=dtype,
            seed=32,
        )


@pytest.mark.parametrize("chunks", [2, 4, 8, 9])
def test_call_ones_taylor(chunks):
    model = orc.drivers.ParallelTaylorDriver(
        n_terms=3,
        res_dim=212,
        spectral_radius=0.6,
        density=0.2,
        bias=1.6,
        dtype=jnp.float64,
        seed=0,
        chunks=chunks,
    )
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key)
    test_vec1 = jax.random.normal(key=key1, shape=(chunks, 212))
    test_vec2 = jax.random.normal(key=key2, shape=(chunks, 212))
    test_outputs = model(test_vec1, test_vec2)

    # Verify output shape
    assert test_outputs.shape == (chunks, 212)

    # Verify outputs are finite
    assert jnp.all(jnp.isfinite(test_outputs))


@pytest.mark.parametrize(
    "res_dim, spectral_radius, density, chunks",
    [
        (100, 0.876, 0.02, 1),
        (500, 0.546, 0.01, 1),
        (1000, 0.432, 0.01, 1),
        (1000, 0.1, 0.01, 1),
        (1000, 1.3, 0.01, 1),
        (100, 0.345, 0.02, 15),
        (500, 0.673, 0.01, 4),
    ],
)
def test_taylordriver_spectral_radius_sparse(res_dim, spectral_radius, density, chunks):
    """Test that the spectral radius of the reservoir update matrix is as expected."""
    driver = orc.drivers.ParallelTaylorDriver(
        n_terms=3,
        res_dim=res_dim,
        spectral_radius=spectral_radius,
        density=density,
        dtype=jnp.float64,
        seed=0,
        chunks=chunks,
        use_sparse_eigs=True,
    )

    wr = driver.wr
    wr_max_eigs = jnp.max(jnp.abs(jnp.linalg.eigvals(wr.todense())), axis=1)
    assert jnp.isclose(wr_max_eigs, spectral_radius, atol=1e-5).all(), (
        f"Expected spectral radius {spectral_radius}, but got {wr_max_eigs}"
    )


@pytest.mark.parametrize(
    "res_dim, spectral_radius, density, chunks",
    [
        (10, 0.6, 0.5, 1),
        (50, 0.6, 0.1, 1),
        (100, 0.876, 0.01, 1),
        (1000, 0.432, 0.01, 1),
        (1000, 0.1, 0.01, 1),
        (1000, 1.3, 0.01, 1),
        (100, 0.345, 0.02, 15),
        (500, 0.673, 0.01, 4),
    ],
)
def test_taylordriver_spectral_radius_dense(res_dim, spectral_radius, density, chunks):
    """Test that the spectral radius of the reservoir update matrix is as expected"""
    driver = orc.drivers.ParallelTaylorDriver(
        n_terms=3,
        res_dim=res_dim,
        spectral_radius=spectral_radius,
        density=density,
        dtype=jnp.float64,
        seed=0,
        chunks=chunks,
        use_sparse_eigs=False,
    )

    wr = driver.wr
    wr_max_eigs = jnp.max(jnp.abs(jnp.linalg.eigvals(wr.todense())), axis=1)
    assert jnp.isclose(wr_max_eigs, spectral_radius, atol=1e-5).all(), (
        f"Expected spectral radius {spectral_radius}, but got {wr_max_eigs}"
    )


@pytest.mark.parametrize("n_terms", [1, 2, 3, 4, 5])
def test_taylordriver_n_terms(n_terms):
    """Test that ParallelTaylorDriver works with different numbers of Taylor terms."""
    driver = orc.drivers.ParallelTaylorDriver(
        n_terms=n_terms,
        res_dim=100,
        spectral_radius=0.6,
        density=0.2,
        bias=1.6,
        dtype=jnp.float64,
        seed=42,
    )

    key = jax.random.key(999)
    test_vec = jax.random.normal(key, shape=(1, 100))
    out_vec = driver.advance(test_vec, test_vec)

    assert out_vec.shape == (1, 100)
    assert jnp.all(jnp.isfinite(out_vec))


@pytest.mark.parametrize(
    "res_dim, spectral_radius, density, chunks, batch_size",
    [
        (100, 0.8, 0.1, 10, 3),
        (150, 0.6, 0.05, 20, 5),
        (200, 0.9, 0.02, 50, 10),
        (100, 0.7, 0.1, 100, 25),
    ],
)
def test_taylordriver_batched_eigenvals_sparse_equivalence(
    res_dim, spectral_radius, density, chunks, batch_size
):
    """Test that batched sparse computation gives same results as non-batched."""
    seed = 42

    # Create driver without batching (default behavior)
    driver_unbatched = orc.drivers.ParallelTaylorDriver(
        n_terms=3,
        res_dim=res_dim,
        spectral_radius=spectral_radius,
        density=density,
        chunks=chunks,
        seed=seed,
        use_sparse_eigs=True,
    )

    # Create driver with batching
    driver_batched = orc.drivers.ParallelTaylorDriver(
        n_terms=3,
        res_dim=res_dim,
        spectral_radius=spectral_radius,
        density=density,
        chunks=chunks,
        seed=seed,
        use_sparse_eigs=True,
        eigenval_batch_size=batch_size,
    )

    # Check that spectral radii are equivalent
    wr_unbatched = driver_unbatched.wr
    wr_batched = driver_batched.wr

    unbatched_eigs = jnp.max(
        jnp.abs(jnp.linalg.eigvals(wr_unbatched.todense())), axis=1
    )
    batched_eigs = jnp.max(jnp.abs(jnp.linalg.eigvals(wr_batched.todense())), axis=1)

    # Both should achieve the target spectral radius
    assert jnp.allclose(unbatched_eigs, spectral_radius, atol=1e-5)
    assert jnp.allclose(batched_eigs, spectral_radius, atol=1e-5)

    # Results should be identical (same seed, same computation)
    assert jnp.allclose(unbatched_eigs, batched_eigs, atol=1e-10)


@pytest.mark.parametrize(
    "res_dim, spectral_radius, density, chunks, batch_size",
    [
        (150, 0.8, 0.1, 10, 3),
        (100, 0.6, 0.05, 20, 7),
        (200, 0.9, 0.02, 30, 8),
    ],
)
def test_taylordriver_batched_eigenvals_dense_equivalence(
    res_dim, spectral_radius, density, chunks, batch_size
):
    """Test that batched dense computation gives same results as non-batched."""
    seed = 123

    # Create driver without batching (default behavior)
    driver_unbatched = orc.drivers.ParallelTaylorDriver(
        n_terms=3,
        res_dim=res_dim,
        spectral_radius=spectral_radius,
        density=density,
        chunks=chunks,
        seed=seed,
        use_sparse_eigs=False,
    )

    # Create driver with batching
    driver_batched = orc.drivers.ParallelTaylorDriver(
        n_terms=3,
        res_dim=res_dim,
        spectral_radius=spectral_radius,
        density=density,
        chunks=chunks,
        seed=seed,
        use_sparse_eigs=False,
        eigenval_batch_size=batch_size,
    )

    # Check that spectral radii are equivalent
    wr_unbatched = driver_unbatched.wr
    wr_batched = driver_batched.wr

    unbatched_eigs = jnp.max(
        jnp.abs(jnp.linalg.eigvals(wr_unbatched.todense())), axis=1
    )
    batched_eigs = jnp.max(jnp.abs(jnp.linalg.eigvals(wr_batched.todense())), axis=1)

    # Both should achieve the target spectral radius
    assert jnp.allclose(unbatched_eigs, spectral_radius, atol=1e-5)
    assert jnp.allclose(batched_eigs, spectral_radius, atol=1e-5)

    # Results should be identical (same seed, same computation)
    assert jnp.allclose(unbatched_eigs, batched_eigs, atol=1e-10)


##################### SINGLE TAYLOR DRIVER TESTS #####################


@pytest.fixture
def single_taylordriver():
    return orc.drivers.TaylorDriver(
        n_terms=3,
        res_dim=100,
        spectral_radius=0.9,
        density=0.1,
        bias=1.0,
        dtype=jnp.float64,
        seed=42,
    )


def test_single_taylordriver_dims(single_taylordriver):
    """Test that TaylorDriver works with single reservoir (no chunks dimension)."""
    key = jax.random.key(123)
    res_dim = single_taylordriver.res_dim

    # Test single state advance
    proj_vars = jax.random.normal(key, shape=(res_dim,))
    res_state = jax.random.normal(key, shape=(res_dim,))
    out_state = single_taylordriver.advance(proj_vars, res_state)

    assert out_state.shape == (res_dim,)
    assert jnp.all(jnp.isfinite(out_state))


def test_single_taylordriver_call(single_taylordriver):
    """Test TaylorDriver __call__ method handles both single and batch inputs."""
    key = jax.random.key(456)
    res_dim = single_taylordriver.res_dim

    # Test single input
    proj_vars = jax.random.normal(key, shape=(res_dim,))
    res_state = jax.random.normal(key, shape=(res_dim,))
    out_state = single_taylordriver(proj_vars, res_state)
    assert out_state.shape == (res_dim,)

    # Test batch input
    batch_proj_vars = jax.random.normal(key, shape=(5, res_dim))
    batch_res_state = jax.random.normal(key, shape=(5, res_dim))
    batch_out = single_taylordriver(batch_proj_vars, batch_res_state)
    assert batch_out.shape == (5, res_dim)


def test_single_taylordriver_chunks_is_one(single_taylordriver):
    """Test that TaylorDriver always has chunks=1."""
    assert single_taylordriver.chunks == 1


##################### GRU DRIVER TESTS #####################


@pytest.fixture
def grudriver():
    return orc.drivers.GRUDriver(
        res_dim=100,
        seed=42,
    )


def test_grudriver_initialization():
    """Test that GRUDriver initializes correctly."""
    driver = orc.drivers.GRUDriver(res_dim=128, seed=0)
    assert driver.res_dim == 128
    assert driver.gru is not None


def test_grudriver_dims(grudriver):
    """Test that GRUDriver works with correct dimensions."""
    key = jax.random.key(123)
    res_dim = grudriver.res_dim

    # Test single state advance
    in_state = jax.random.normal(key, shape=(res_dim,))
    res_state = jax.random.normal(key, shape=(res_dim,))
    out_state = grudriver.advance(res_state, in_state)

    assert out_state.shape == (res_dim,)
    assert jnp.all(jnp.isfinite(out_state))


def test_grudriver_reproducibility():
    """Test that GRUDriver produces reproducible results with same seed."""
    key = jax.random.key(456)
    in_state = jax.random.normal(key, shape=(100,))
    res_state = jax.random.normal(key, shape=(100,))

    # Create two drivers with same seed
    driver1 = orc.drivers.GRUDriver(res_dim=100, seed=42)
    driver2 = orc.drivers.GRUDriver(res_dim=100, seed=42)

    out1 = driver1.advance(res_state, in_state)
    out2 = driver2.advance(res_state, in_state)

    assert jnp.allclose(out1, out2)


def test_grudriver_different_seeds():
    """Test that different seeds produce different drivers."""
    key = jax.random.key(789)
    in_state = jax.random.normal(key, shape=(100,))
    res_state = jax.random.normal(key, shape=(100,))

    driver1 = orc.drivers.GRUDriver(res_dim=100, seed=42)
    driver2 = orc.drivers.GRUDriver(res_dim=100, seed=123)

    out1 = driver1.advance(res_state, in_state)
    out2 = driver2.advance(res_state, in_state)

    # Should produce different outputs due to different initialization
    assert not jnp.allclose(out1, out2)


@pytest.mark.parametrize("batch_size", [3, 10, 25])
def test_batchapply_dims_gru(batch_size, grudriver):
    """Test batch advance functionality for GRUDriver."""
    key = jax.random.key(42)
    res_dim = grudriver.res_dim
    in_states = jax.random.normal(key, shape=(batch_size, res_dim))
    res_states = jax.random.normal(key, shape=(batch_size, res_dim))

    out_states = grudriver.batch_advance(res_states, in_states)

    assert out_states.shape == (batch_size, res_dim)
    assert jnp.all(jnp.isfinite(out_states))


def test_grudriver_call_single(grudriver):
    """Test GRUDriver __call__ method with single input."""
    key = jax.random.key(111)
    res_dim = grudriver.res_dim

    in_state = jax.random.normal(key, shape=(res_dim,))
    res_state = jax.random.normal(key, shape=(res_dim,))
    out_state = grudriver(res_state, in_state)

    assert out_state.shape == (res_dim,)
    assert jnp.all(jnp.isfinite(out_state))


def test_grudriver_call_batch(grudriver):
    """Test GRUDriver __call__ method with batch input."""
    key = jax.random.key(222)
    res_dim = grudriver.res_dim
    batch_size = 7

    in_states = jax.random.normal(key, shape=(batch_size, res_dim))
    res_states = jax.random.normal(key, shape=(batch_size, res_dim))
    out_states = grudriver.batch_advance(res_states, in_states)

    assert out_states.shape == (batch_size, res_dim)
    assert jnp.all(jnp.isfinite(out_states))


def test_grudriver_stateful_behavior():
    """Test that GRUDriver properly updates state across multiple steps."""
    driver = orc.drivers.GRUDriver(res_dim=50, seed=0)
    key = jax.random.key(333)

    # Initialize states
    res_state = jax.random.normal(key, shape=(50,))
    in_state = jax.random.normal(key, shape=(50,))

    # Advance multiple steps
    state1 = driver.advance(res_state, in_state)
    state2 = driver.advance(state1, in_state)
    state3 = driver.advance(state2, in_state)

    # Each state should be different (GRU has memory)
    assert not jnp.allclose(state1, state2)
    assert not jnp.allclose(state2, state3)
    assert not jnp.allclose(state1, state3)

    # All states should be finite
    assert jnp.all(jnp.isfinite(state1))
    assert jnp.all(jnp.isfinite(state2))
    assert jnp.all(jnp.isfinite(state3))


def test_grudriver_param_types():
    """Test that GRUDriver raises errors for invalid parameter types."""
    # res_dim must be an integer
    with pytest.raises(TypeError):
        _ = orc.drivers.GRUDriver(res_dim=100.5, seed=0)


def test_grudriver_consistency_with_equinox_grucell():
    """Test that GRUDriver advance matches direct GRUCell usage."""
    res_dim = 80
    seed = 99

    # Create driver
    driver = orc.drivers.GRUDriver(res_dim=res_dim, seed=seed)

    # Create test data
    test_key = jax.random.key(1000)
    in_state = jax.random.normal(test_key, shape=(res_dim,))
    res_state = jax.random.normal(test_key, shape=(res_dim,))

    # Get output from driver
    driver_output = driver.advance(res_state, in_state)

    # Get output from direct GRU cell usage
    direct_output = driver.gru(in_state, res_state)

    # Should be identical
    assert jnp.allclose(driver_output, direct_output)
