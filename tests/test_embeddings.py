import jax
import jax.numpy as jnp
import pytest

import orc


@pytest.fixture
def linearembedding():
    return orc.embeddings.LinearEmbedding(
        in_dim=3, res_dim=982, scaling=0.2745, dtype=jnp.float64, seed=0
    )


def test_linearembedding_dims(linearembedding):
    key = jax.random.key(999)
    in_dim = linearembedding.in_dim
    res_dim = linearembedding.res_dim
    test_vec = jax.random.normal(key, shape=(in_dim))
    out_vec = linearembedding.embed(test_vec)
    assert out_vec.shape == (res_dim,)

    test_vec = jax.random.normal(key, shape=(in_dim - 1))
    with pytest.raises(ValueError):
        out_vec = linearembedding.embed(test_vec)


@pytest.mark.parametrize("batch_size", [3, 12, 52])
def test_batchapply_dims_linear(batch_size, linearembedding):
    key = jax.random.key(42)
    in_dim = linearembedding.in_dim
    res_dim = linearembedding.res_dim
    test_vec = jax.random.normal(key, shape=(batch_size, in_dim))
    out_vec = linearembedding.batch_embed(test_vec)

    assert out_vec.shape == (batch_size, res_dim)

    test_vec = jax.random.normal(key, shape=(batch_size, in_dim - 1))

    with pytest.raises(ValueError):
        out_vec = linearembedding.batch_embed(test_vec)


@pytest.mark.parametrize(
    "in_dim,res_dim,scaling,dtype",
    [
        (2, 230.2, 2, jnp.float64),
        (3.1, 230, 3.2, jnp.float32),
        (3, 222, 0.084, jnp.int32),
    ],
)
def test_param_types_linearembedding(in_dim, res_dim, scaling, dtype):
    with pytest.raises(TypeError):
        _ = orc.embeddings.LinearEmbedding(
            in_dim=in_dim,
            res_dim=res_dim,
            scaling=scaling,
            dtype=dtype,
            seed=111,
        )
