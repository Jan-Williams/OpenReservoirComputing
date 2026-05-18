import equinox as eqx
import jax
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


##################### ENSEMBLE LINEAR EMBEDDING TESTS #####################


@pytest.mark.parametrize(
    "chunks,batch_size,in_dim",
    [
        (5, 32, 3),
        (3, 16, 4),
        (15, 17, 5),
    ],
)
def test_ensemble_embed_shapes(chunks, batch_size, in_dim):
    res_dim = 312
    in_dim = 5
    scaling = 0.084
    embedding = orc.embeddings.EnsembleLinearEmbedding(
        in_dim, res_dim, scaling, chunks, seed=0
    )
    inputs = jnp.ones((batch_size, in_dim))
    outputs = embedding(inputs)
    assert outputs.shape == (batch_size, chunks, res_dim)

    inputs = jnp.ones(in_dim)
    outputs = embedding(inputs)
    assert outputs.shape == (chunks, res_dim)


##################### JAX TRANSFORM & AD TESTS #####################


@pytest.fixture
def parallel_linearembedding():
    return orc.embeddings.ParallelLinearEmbedding(
        in_dim=16,
        res_dim=64,
        scaling=0.1,
        chunks=4,
        locality=1,
        dtype=jnp.float64,
        seed=0,
    )


@pytest.fixture
def ensemble_linearembedding():
    return orc.embeddings.EnsembleLinearEmbedding(
        in_dim=5,
        res_dim=64,
        scaling=0.1,
        chunks=4,
        dtype=jnp.float64,
        seed=0,
    )


@pytest.mark.parametrize(
    "embedding_fixture",
    [
        "single_linearembedding",
        "parallel_linearembedding",
        "ensemble_linearembedding",
    ],
)
def test_embedding_transform_stability(embedding_fixture, request):
    """Verify embeddings are compatible with vmap and jit."""
    embedding = request.getfixturevalue(embedding_fixture)
    key = jax.random.key(999)
    batch_size = 3

    batch_in = jax.random.normal(key, shape=(batch_size, embedding.in_dim))

    vmap_embed = eqx.filter_vmap(embedding.embed)
    assert jnp.allclose(
        vmap_embed(batch_in),
        embedding.batch_embed(batch_in),
    )

    jit_embed = eqx.filter_jit(embedding.embed)
    assert jnp.all(jnp.isfinite(jit_embed(batch_in[0])))


@pytest.mark.parametrize(
    "embedding_fixture",
    [
        "single_linearembedding",
        "parallel_linearembedding",
        "ensemble_linearembedding",
    ],
)
def test_embedding_differentiability(embedding_fixture, request):
    """Verify gradients flow backward through the embed step via Equinox."""
    embedding = request.getfixturevalue(embedding_fixture)
    key = jax.random.key(999)

    in_state = jax.random.normal(key, shape=(embedding.in_dim,))

    @eqx.filter_value_and_grad
    def loss_fn(model, x):
        return jnp.sum(model.embed(x))

    loss, grads = loss_fn(embedding, in_state)
    assert jnp.isfinite(loss)

    def check_finite(g):
        if eqx.is_array(g) and jnp.issubdtype(g.dtype, jnp.inexact):
            assert jnp.all(jnp.isfinite(g))

    jax.tree_util.tree_map(check_finite, grads)
