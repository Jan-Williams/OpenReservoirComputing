import jax
import jax.numpy as jnp
import pytest
from jax.experimental import sparse

import orc


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
     ])
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
     ])
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
