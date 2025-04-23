import jax
import jax.numpy as jnp
import pytest

import orc


@pytest.mark.parametrize(
    "in_dim,groups,locality", [(16, 8, 2), (32, 4, 1), (22, 11, 3), (14, 1, 0)]
)
def test_win_dims_Linear(in_dim, groups, locality):
    model = orc.embeddings.LinearEmbedding(
        in_dim=in_dim,
        res_dim=200,
        scaling=0.014,
        dtype=jnp.float32,
        seed=0,
        groups=groups,
        locality=locality,
    )
    assert model.win.shape == (groups, 200, int(in_dim / groups) + 2 * locality)


@pytest.mark.parametrize(
    "in_dim,groups,locality", [(16, 7, 2), (32, 3, 1), (22, 12, 3)]
)
def test_bad_group_nums_Linear(in_dim, groups, locality):
    with pytest.raises(ValueError):
        model = orc.embeddings.LinearEmbedding(
            in_dim=in_dim,
            res_dim=200,
            scaling=0.014,
            dtype=jnp.float32,
            seed=0,
            groups=groups,
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
        _ = orc.embeddings.LinearEmbedding(
            in_dim=in_dim,
            res_dim=res_dim,
            scaling=scaling,
            dtype=dtype,
            seed=111,
        )


@pytest.mark.parametrize(
    "groups,locality,seq_len,",
    [
        (5, 2, 20),
        (3, 12, 1),
        (15, 10, 30),
    ],
)
def test_call_Linear(groups, locality, seq_len):
    model = orc.embeddings.LinearEmbedding(
        in_dim=180,
        res_dim=300,
        scaling=0.12,
        locality=locality,
        groups=groups,
        seed=123,
    )
    output = model(jnp.ones((seq_len, 180)))
    assert output.shape == (seq_len, groups, 300)
