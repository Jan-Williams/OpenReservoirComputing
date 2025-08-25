import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.experimental import sparse

import orc

##################### ESN TESTS #####################


@pytest.fixture
def esndriver():
    return orc.drivers.ESNDriver(
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
        _ = orc.drivers.ESNDriver(
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
        _ = orc.drivers.ESNDriver(
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
    model = orc.drivers.ESNDriver(
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
    driver = orc.drivers.ESNDriver(
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
    driver = orc.drivers.ESNDriver(
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
    driver_unbatched = orc.drivers.ESNDriver(
        res_dim=res_dim,
        spectral_radius=spectral_radius,
        density=density,
        chunks=chunks,
        seed=seed,
        use_sparse_eigs=True,
    )

    # Create driver with batching
    driver_batched = orc.drivers.ESNDriver(
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
    driver_unbatched = orc.drivers.ESNDriver(
        res_dim=res_dim,
        spectral_radius=spectral_radius,
        density=density,
        chunks=chunks,
        seed=seed,
        use_sparse_eigs=False,
    )

    # Create driver with batching
    driver_batched = orc.drivers.ESNDriver(
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

    driver = orc.drivers.ESNDriver(
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
    driver = orc.drivers.ESNDriver(
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
    driver = orc.drivers.ESNDriver(
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
    driver = orc.drivers.ESNDriver(
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


##################### CESN TESTS #####################


@pytest.fixture
def cesn_driver():
    return orc.drivers.ESNDriver(
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
        _ = orc.drivers.ESNDriver(
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
        _ = orc.drivers.ESNDriver(
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
    model = orc.drivers.ESNDriver(
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


##################### ParGRUCell TESTS #####################


@pytest.fixture
def pargru_cell():
    return orc.drivers._ParGRUCell(
        input_size=10,
        hidden_size=20,
        chunks=3,
        use_bias=True,
        seed=42,
        dtype=jnp.float64,
    )


def test_pargru_cell_initialization(pargru_cell):
    """Test that _ParGRUCell initializes correctly."""
    assert pargru_cell.input_size == 10
    assert pargru_cell.hidden_size == 20
    assert pargru_cell.chunks == 3
    assert pargru_cell.use_bias is True
    assert pargru_cell.dtype == jnp.float64

    assert pargru_cell.weight_ih.shape == (3, 3 * 20, 10)
    assert pargru_cell.weight_hh.shape == (3, 3 * 20, 20)

    assert pargru_cell.bias.shape == (3, 3 * 20)
    assert pargru_cell.bias_n.shape == (3, 20)


def test_pargru_cell_no_bias():
    """Test _ParGRUCell with bias disabled."""
    cell = orc.drivers._ParGRUCell(
        input_size=5,
        hidden_size=10,
        chunks=2,
        use_bias=False,
        seed=123,
        dtype=jnp.float32,
    )

    assert cell.use_bias is False
    assert cell.bias is None
    assert cell.bias_n is None
    assert cell.dtype == jnp.float32


@pytest.mark.parametrize(
    "input_size,hidden_size,chunks",
    [
        (5, 10, 1),
        (10, 20, 2),
        (15, 25, 4),
        (32, 64, 8),
    ],
)
def test_pargru_cell_forward_shapes(input_size, hidden_size, chunks):
    """Test that forward pass produces correct output shapes."""
    cell = orc.drivers._ParGRUCell(
        input_size=input_size,
        hidden_size=hidden_size,
        chunks=chunks,
        seed=42,
    )

    key = jax.random.key(999)
    input_data = jax.random.normal(key, shape=(chunks, input_size))
    hidden_data = jax.random.normal(key, shape=(chunks, hidden_size))

    output = cell(input_data, hidden_data)

    print(output)
    assert output.shape == (chunks, hidden_size)
    assert jnp.all(jnp.isfinite(output))


##################### GRU DRIVER TESTS #####################


@pytest.fixture
def grudriver():
    return orc.drivers.GRUDriver(
        res_dim=212,
        chunks=1,
        seed=0,
        use_bias=True,
    )


def test_grudriver_dims(grudriver):
    key = jax.random.key(999)
    res_dim = grudriver.res_dim
    test_vec = jax.random.normal(key, shape=(1, res_dim))
    out_vec = grudriver.advance(test_vec, test_vec)
    assert out_vec.shape == (
        1,
        res_dim,
    )


@pytest.mark.parametrize("batch_size", [3, 12, 52])
def test_batchapply_dims_gru(batch_size, grudriver):
    key = jax.random.key(42)
    res_dim = grudriver.res_dim
    test_vec = jax.random.normal(key, shape=(batch_size, 1, res_dim))
    out_vec = grudriver.batch_advance(test_vec, test_vec)

    assert out_vec.shape == (batch_size, 1, res_dim)


@pytest.mark.parametrize("chunks", [2, 4, 8, 9])
def test_call_ones_gru(chunks):
    model = orc.drivers.GRUDriver(
        res_dim=212,
        chunks=chunks,
        seed=0,
        use_bias=True,
    )
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key)
    test_vec1 = jax.random.normal(key=key1, shape=(chunks, 212))
    test_vec2 = jax.random.normal(key=key2, shape=(chunks, 212))
    test_outputs = model(test_vec1, test_vec2)

    assert test_outputs.shape == (chunks, 212)
    assert jnp.all(jnp.isfinite(test_outputs))

    gru_cell = model.gru
    manual_output = gru_cell(test_vec2, test_vec1)
    assert jnp.allclose(test_outputs, manual_output)


@pytest.mark.parametrize("use_bias", [True, False])
def test_gru_bias_handling(use_bias):
    model = orc.drivers.GRUDriver(
        res_dim=100,
        chunks=2,
        seed=42,
        use_bias=use_bias,
    )

    key = jax.random.key(123)
    proj_vars = jax.random.normal(key, shape=(2, 100))
    res_state = jax.random.normal(key, shape=(2, 100))

    output = model.advance(res_state, proj_vars)

    assert output.shape == (2, 100)
    assert jnp.all(jnp.isfinite(output))
    assert model.gru.use_bias == use_bias


@pytest.mark.parametrize("res_dim", [50, 100, 200, 512])
def test_gru_different_dimensions(res_dim):
    model = orc.drivers.GRUDriver(
        res_dim=res_dim,
        chunks=1,
        seed=777,
    )

    key = jax.random.key(888)
    proj_vars = jax.random.normal(key, shape=(1, res_dim))
    res_state = jax.random.normal(key, shape=(1, res_dim))

    output = model.advance(res_state, proj_vars)

    assert output.shape == (1, res_dim)
    assert jnp.all(jnp.isfinite(output))


def test_gru_deterministic():
    """Test that same inputs produce same outputs."""
    model1 = orc.drivers.GRUDriver(res_dim=100, chunks=2, seed=999)
    model2 = orc.drivers.GRUDriver(res_dim=100, chunks=2, seed=999)

    key = jax.random.key(111)
    proj_vars = jax.random.normal(key, shape=(2, 100))
    res_state = jax.random.normal(key, shape=(2, 100))

    output1 = model1.advance(res_state, proj_vars)
    output2 = model2.advance(res_state, proj_vars)

    assert jnp.allclose(output1, output2)


def test_gru_gradient_flow():
    """Test that gradients flow through the GRU driver correctly."""
    model = orc.drivers.GRUDriver(res_dim=50, chunks=1, seed=42)

    key = jax.random.key(123)
    proj_vars = jax.random.normal(key, shape=(1, 50))
    res_state = jax.random.normal(key, shape=(1, 50))

    def loss_fn(model, res_state, proj_vars):
        output = model.advance(res_state, proj_vars)
        return jnp.sum(output**2)

    grad_fn = eqx.filter_grad(loss_fn)
    grads = grad_fn(model, res_state, proj_vars)

    assert hasattr(grads, "gru")
    assert jnp.all(jnp.isfinite(grads.gru.weight_ih))
    assert jnp.all(jnp.isfinite(grads.gru.weight_hh))

    if model.gru.use_bias:
        assert jnp.all(jnp.isfinite(grads.gru.bias))
        assert jnp.all(jnp.isfinite(grads.gru.bias_n))

    def loss_fn(proj_vars, model, res_state):
        output = model.advance(res_state, proj_vars)
        return jnp.sum(output**2)

    grad_fn = eqx.filter_grad(loss_fn)
    grads = grad_fn(proj_vars, model, res_state)
    assert grads.shape == proj_vars.shape


@pytest.mark.parametrize("chunks", [1, 3, 5, 8])
def test_gru_parallel_chunks_functionality(chunks):
    """Test GRU driver functionality with different chunk sizes."""
    model = orc.drivers.GRUDriver(
        res_dim=64,
        chunks=chunks,
        seed=555,
    )

    key = jax.random.key(666)
    proj_vars = jax.random.normal(key, shape=(chunks, 64))
    res_state = jax.random.normal(key, shape=(chunks, 64))

    output = model.advance(res_state, proj_vars)
    assert output.shape == (chunks, 64)
    assert jnp.all(jnp.isfinite(output))

    batch_proj_vars = jax.random.normal(key, shape=(5, chunks, 64))
    batch_res_state = jax.random.normal(key, shape=(5, chunks, 64))

    batch_output = model.batch_advance(batch_res_state, batch_proj_vars)
    assert batch_output.shape == (5, chunks, 64)
    assert jnp.all(jnp.isfinite(batch_output))

    call_output = model(batch_res_state, batch_proj_vars)
    assert jnp.allclose(call_output, batch_output)
