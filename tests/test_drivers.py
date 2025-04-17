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
        spec_rad=0.6,
        density=0.02,
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
    "res_dim,leak,spec_rad,density,bias,dtype",
    [
        (22, 0.123, 0.6, 0.02, 1.6, jnp.int32),
        (22.2, 0.123, 0.6, 0.02, 1.6, jnp.float64),
    ],
)
def test_param_types_esn(res_dim, leak, spec_rad, density, bias, dtype):
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
def test_param_vals_esn(res_dim, leak, spec_rad, density, bias, dtype):
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


@pytest.mark.parametrize("groups", [2, 4, 8, 9])
def test_call_ones_esn(groups):
    model = orc.drivers.ESNDriver(
        res_dim=212,
        leak=0.123,
        spec_rad=0.6,
        density=0.02,
        bias=1.6,
        dtype=jnp.float64,
        seed=0,
        groups=groups,
    )
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key)
    test_vec1 = jax.random.normal(key=key1, shape=(groups, 212))
    test_vec2 = jax.random.normal(key=key2, shape=(groups, 212))
    test_outputs = model(test_vec1, test_vec2)
    wr = sparse.BCOO.todense(model.wr)
    bias = model.bias
    leak = model.leak

    def naive_imp_forward(wr, bias, leak, proj_vars, res_state):
        res_next = jnp.tanh(
            wr @ res_state + proj_vars + bias * jnp.ones_like(proj_vars)
        )
        return leak * res_next + (1 - leak) * res_state

    gt_outputs = jnp.empty((groups, 212))
    for group in range(groups):
        gt = naive_imp_forward(
            wr[group], bias, leak, test_vec1[group], test_vec2[group]
        )
        gt_outputs = gt_outputs.at[group].set(gt)
    assert jnp.allclose(gt_outputs, test_outputs)
