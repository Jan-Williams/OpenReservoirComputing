import jax
import jax.numpy as jnp
import pytest

import orc


@pytest.fixture
def esndriver():
    return orc.drivers.ESNDriver(
        res_dim=212,
        leak=0.123,
        spec_rad=0.6,
        density=0.02,
        bias=1.6,
        dtype=jnp.float64,
        seed=0,
    )


def test_esndriver_dims(esndriver):
    key = jax.random.key(999)
    res_dim = esndriver.res_dim
    test_vec = jax.random.normal(key, shape=(res_dim))
    out_vec = esndriver.advance(test_vec, test_vec)
    assert out_vec.shape == (res_dim,)

    test_vec = jax.random.normal(key, shape=(res_dim - 1))
    with pytest.raises(ValueError):
        out_vec = esndriver.advance(test_vec, test_vec)


@pytest.mark.parametrize("batch_size", [3, 12, 52])
def test_batchapply_dims_esn(batch_size, esndriver):
    key = jax.random.key(42)
    res_dim = esndriver.res_dim
    test_vec = jax.random.normal(key, shape=(batch_size, res_dim))
    out_vec = esndriver.batch_advance(test_vec, test_vec)

    assert out_vec.shape == (batch_size, res_dim)

    test_vec = jax.random.normal(key, shape=(batch_size, res_dim - 1))

    with pytest.raises(ValueError):
        out_vec = esndriver.batch_advance(test_vec, test_vec)


@pytest.mark.parametrize(
    "res_dim,leak,spec_rad,density,bias,dtype",
    [
        (22, 0.123, 0.6, 0.02, 1.6, jnp.int32),
        (22.2, 0.123, 0.6, 0.02, 1.6, jnp.float64),
    ],
)
def test_param_types_linearreadout(res_dim, leak, spec_rad, density, bias, dtype):
    with pytest.raises(TypeError):
        _ = orc.drivers.ESNDriver(
            res_dim=res_dim,
            leak=leak,
            spec_rad=spec_rad,
            density=density,
            bias=bias,
            dtype=dtype,
            seed=33,
        )


@pytest.mark.parametrize(
    "res_dim,leak,spec_rad,density,bias,dtype",
    [
        (22, 0.123, -0.5, 0.02, 1.6, jnp.float32),
        (22, 0.123, 0.6, 1.3, 1.6, jnp.float64),
        (22, -0.2, 0.6, 0.04, 1.6, jnp.float32),
    ],
)
def test_param_vals_linearreadout(res_dim, leak, spec_rad, density, bias, dtype):
    with pytest.raises(ValueError):
        _ = orc.drivers.ESNDriver(
            res_dim=res_dim,
            leak=leak,
            spec_rad=spec_rad,
            density=density,
            bias=bias,
            dtype=dtype,
            seed=32,
        )
