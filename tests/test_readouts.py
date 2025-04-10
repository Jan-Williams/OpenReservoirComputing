import jax
import jax.numpy as jnp
import pytest

import orc


@pytest.fixture
def linearreadout():
    return orc.readouts.LinearReadout(out_dim=3, res_dim=982, dtype=jnp.float64, seed=0)


def test_linearreadout_dims(linearreadout):
    key = jax.random.key(999)
    out_dim = linearreadout.out_dim
    res_dim = linearreadout.res_dim
    test_vec = jax.random.normal(key, shape=(res_dim))
    out_vec = linearreadout.readout(test_vec)
    assert out_vec.shape == (out_dim,)

    test_vec = jax.random.normal(key, shape=(res_dim - 1))
    with pytest.raises(ValueError):
        out_vec = linearreadout.readout(test_vec)


@pytest.mark.parametrize("batch_size", [3, 12, 52])
def test_batchapply_dims_linear(batch_size, linearreadout):
    key = jax.random.key(42)
    out_dim = linearreadout.out_dim
    res_dim = linearreadout.res_dim
    test_vec = jax.random.normal(key, shape=(batch_size, res_dim))
    out_vec = linearreadout.batch_readout(test_vec)

    assert out_vec.shape == (batch_size, out_dim)

    test_vec = jax.random.normal(key, shape=(batch_size, res_dim - 1))

    with pytest.raises(ValueError):
        out_vec = linearreadout.batch_readout(test_vec)


@pytest.mark.parametrize(
    "out_dim,res_dim,dtype",
    [(2, 230.2, jnp.float64), (3.1, 230, jnp.float32), (3, 222, jnp.int32)],
)
def test_param_types_linearreadout(out_dim, res_dim, dtype):
    with pytest.raises(TypeError):
        _ = orc.readouts.LinearReadout(
            out_dim=out_dim,
            res_dim=res_dim,
            dtype=dtype,
            seed=111,
        )
