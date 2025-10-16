import jax.numpy as jnp
import pytest

import orc


@pytest.mark.parametrize(
    "in_dim,chunks,locality", [(16, 8, 2), (32, 4, 1), (22, 11, 3), (14, 1, 0)]
)
def test_win_dims_Linear(in_dim, chunks, locality):
    model = orc.embeddings.ParallelLinearEmbedding(
        in_dim=in_dim,
        res_dim=200,
        scaling=0.014,
        dtype=jnp.float32,
        seed=0,
        chunks=chunks,
        locality=locality,
    )
    assert model.win.shape == (chunks, 200, int(in_dim / chunks) + 2 * locality)


@pytest.mark.parametrize(
    "in_dim,chunks,locality", [(16, 7, 2), (32, 3, 1), (22, 12, 3)]
)
def test_bad_group_nums_Linear(in_dim, chunks, locality):
    with pytest.raises(ValueError):
        _ = orc.embeddings.ParallelLinearEmbedding(
            in_dim=in_dim,
            res_dim=200,
            scaling=0.014,
            dtype=jnp.float32,
            seed=0,
            chunks=chunks,
            locality=locality,
        )


@pytest.mark.parametrize(
    "in_dim,res_dim,scaling,dtype",
    [
        (2, 230.2, 2, jnp.float64),
        (3.1, 230, 3.2, jnp.float32),
        (3, 222, 0.084, jnp.int32),
    ],
)
def test_param_types_Linear(in_dim, res_dim, scaling, dtype):
    with pytest.raises(TypeError):
        _ = orc.embeddings.ParallelLinearEmbedding(
            in_dim=in_dim,
            res_dim=res_dim,
            scaling=scaling,
            dtype=dtype,
            seed=111,
        )


@pytest.mark.parametrize(
    "chunks,locality,seq_len,",
    [
        (5, 2, 20),
        (3, 12, 1),
        (15, 10, 30),
    ],
)
def test_call_Linear(chunks, locality, seq_len):
    model = orc.embeddings.ParallelLinearEmbedding(
        in_dim=180,
        res_dim=300,
        scaling=0.12,
        locality=locality,
        chunks=chunks,
        seed=123,
    )
    output = model(jnp.ones((seq_len, 180)))
    assert output.shape == (seq_len, chunks, 300)


##################### SINGLE LINEAR EMBEDDING TESTS #####################


@pytest.fixture
def single_linearembedding():
    return orc.embeddings.LinearEmbedding(
        in_dim=50,
        res_dim=100,
        scaling=0.1,
        dtype=jnp.float64,
        seed=42,
    )


def test_single_linearembedding_dims(single_linearembedding):
    """Test that LinearEmbedding works with single embedding (no chunks dimension)."""
    in_dim = single_linearembedding.in_dim
    res_dim = single_linearembedding.res_dim

    # Test single state embed
    in_state = jnp.ones(in_dim)
    out_state = single_linearembedding.embed(in_state)

    assert out_state.shape == (res_dim,)
    assert jnp.all(jnp.isfinite(out_state))


def test_single_linearembedding_call(single_linearembedding):
    """Test LinearEmbedding __call__ method handles both single and batch inputs."""
    in_dim = single_linearembedding.in_dim
    res_dim = single_linearembedding.res_dim

    # Test single input
    in_state = jnp.ones(in_dim)
    out_state = single_linearembedding(in_state)
    assert out_state.shape == (res_dim,)

    # Test batch input
    batch_in = jnp.ones((5, in_dim))
    batch_out = single_linearembedding(batch_in)
    assert batch_out.shape == (5, res_dim)


def test_single_linearembedding_chunks_is_one(single_linearembedding):
    """Test that LinearEmbedding always has chunks=1."""
    assert single_linearembedding.chunks == 1

