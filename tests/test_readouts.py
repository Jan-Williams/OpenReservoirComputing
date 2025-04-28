import equinox as eqx
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
    chunks = linearreadout.chunks
    test_vec = jax.random.normal(key, shape=(chunks, res_dim))
    out_vec = linearreadout.readout(test_vec)
    assert out_vec.shape == (out_dim,)

    test_vec = jax.random.normal(key, shape=(chunks, res_dim - 1))
    with pytest.raises(ValueError):
        out_vec = linearreadout.readout(test_vec)


@pytest.mark.parametrize("batch_size", [3, 12, 52])
def test_batchapply_dims_linear(batch_size, linearreadout):
    key = jax.random.key(42)
    out_dim = linearreadout.out_dim
    res_dim = linearreadout.res_dim
    chunks = linearreadout.chunks
    test_vec = jax.random.normal(key, shape=(batch_size, chunks, res_dim))
    out_vec = linearreadout.batch_readout(test_vec)

    assert out_vec.shape == (batch_size, out_dim)

    test_vec = jax.random.normal(key, shape=(batch_size, chunks, res_dim - 1))

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


def test_ravel():
    model = orc.readouts.LinearReadout(
        out_dim=3,
        res_dim=10,
        dtype=jnp.float64,
        chunks=5,
        seed=111,
    )

    def where(m):
        return m.wout

    # to_rep = jnp.repeat(jnp.eye, 32, axis=0).reshape(5,3,10, order='F')
    to_rep = jnp.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    repped = jnp.repeat(to_rep, 5, axis=0).reshape(5, 3, 10, order="F")
    model = eqx.tree_at(where, model, repped)
    test_input = jnp.arange(50).reshape(5, 10)
    test_output = model(test_input)
    assert (
        test_output
        == jnp.array([0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42])
    ).all()
