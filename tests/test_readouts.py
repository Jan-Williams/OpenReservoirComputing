import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import orc


@pytest.fixture
def linearreadout():
    return orc.readouts.ParallelLinearReadout(
        out_dim=3, res_dim=982, dtype=jnp.float64, seed=0
    )


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
        _ = orc.readouts.ParallelLinearReadout(
            out_dim=out_dim,
            res_dim=res_dim,
            dtype=dtype,
            seed=111,
        )


def test_ravel():
    model = orc.readouts.ParallelLinearReadout(
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


def test_quadratic_readout():
    readout = orc.readouts.ParallelNonlinearReadout(
        out_dim=6, res_dim=6, nonlin_list=[lambda x: x**2], chunks=1, dtype=jnp.float64
    )

    def where_mat(readout):
        return readout.wout

    readout = eqx.tree_at(where_mat, readout, jnp.eye(6).reshape(1, 6, 6))
    to_output = jnp.arange(6).reshape(1, -1)
    target_output = jnp.array([0, 1, 2, 9, 4, 25])
    test_out = readout(to_output)
    assert jnp.allclose(test_out, target_output)


def test_nonlin_and_quadratic_readout():
    readout = orc.readouts.ParallelNonlinearReadout(
        out_dim=6, res_dim=6, nonlin_list=[lambda x: x**2], chunks=12, dtype=jnp.float64
    )
    quad_readout = orc.readouts.ParallelQuadraticReadout(
        out_dim=6, res_dim=6, chunks=12, dtype=jnp.float64
    )

    def where_mat(readout):
        return readout.wout

    random_mat = jax.random.normal(jax.random.key(0), shape=(12, 6, 6))
    readout = eqx.tree_at(where_mat, readout, random_mat)
    quad_readout = eqx.tree_at(where_mat, quad_readout, random_mat)
    rand_res_state = jax.random.normal(jax.random.key(0), shape=(12, 6))
    output_1 = readout(rand_res_state)
    output_2 = quad_readout(rand_res_state)
    assert jnp.allclose(output_1, output_2)


##################### SINGLE LINEAR READOUT TESTS #####################


@pytest.fixture
def single_linearreadout():
    return orc.readouts.LinearReadout(
        out_dim=3,
        res_dim=100,
        dtype=jnp.float64,
        seed=42,
    )


def test_single_linearreadout_dims(single_linearreadout):
    """Test that LinearReadout works with single reservoir (no chunks dimension)."""
    key = jax.random.key(123)
    out_dim = single_linearreadout.out_dim
    res_dim = single_linearreadout.res_dim

    # Test single state readout
    res_state = jax.random.normal(key, shape=(res_dim,))
    out_state = single_linearreadout.readout(res_state)

    assert out_state.shape == (out_dim,)
    assert jnp.all(jnp.isfinite(out_state))


def test_single_linearreadout_call(single_linearreadout):
    """Test LinearReadout __call__ method handles both single and batch inputs."""
    key = jax.random.key(456)
    out_dim = single_linearreadout.out_dim
    res_dim = single_linearreadout.res_dim

    # Test single input
    res_state = jax.random.normal(key, shape=(res_dim,))
    out_state = single_linearreadout(res_state)
    assert out_state.shape == (out_dim,)

    # Test batch input
    batch_res_state = jax.random.normal(key, shape=(5, res_dim))
    batch_out = single_linearreadout(batch_res_state)
    assert batch_out.shape == (5, out_dim)


def test_single_linearreadout_chunks_is_one(single_linearreadout):
    """Test that LinearReadout always has chunks=1."""
    assert single_linearreadout.chunks == 1


##################### SINGLE NONLINEAR READOUT TESTS #####################


@pytest.fixture
def single_nonlinearreadout():
    return orc.readouts.NonlinearReadout(
        out_dim=3,
        res_dim=100,
        nonlin_list=[lambda x: x**2],
        dtype=jnp.float64,
        seed=42,
    )


def test_single_nonlinearreadout_dims(single_nonlinearreadout):
    """Test that NonlinearReadout works with single reservoir (no chunks dimension)."""
    key = jax.random.key(123)
    out_dim = single_nonlinearreadout.out_dim
    res_dim = single_nonlinearreadout.res_dim

    # Test single state readout
    res_state = jax.random.normal(key, shape=(res_dim,))
    out_state = single_nonlinearreadout.readout(res_state)

    assert out_state.shape == (out_dim,)
    assert jnp.all(jnp.isfinite(out_state))


def test_single_nonlinearreadout_call(single_nonlinearreadout):
    """Test NonlinearReadout __call__ method handles both single and batch inputs."""
    key = jax.random.key(456)
    out_dim = single_nonlinearreadout.out_dim
    res_dim = single_nonlinearreadout.res_dim

    # Test single input
    res_state = jax.random.normal(key, shape=(res_dim,))
    out_state = single_nonlinearreadout(res_state)
    assert out_state.shape == (out_dim,)

    # Test batch input
    batch_res_state = jax.random.normal(key, shape=(5, res_dim))
    batch_out = single_nonlinearreadout(batch_res_state)
    assert batch_out.shape == (5, out_dim)


def test_single_nonlinearreadout_chunks_is_one(single_nonlinearreadout):
    """Test that NonlinearReadout always has chunks=1."""
    assert single_nonlinearreadout.chunks == 1


##################### SINGLE QUADRATIC READOUT TESTS #####################


@pytest.fixture
def single_quadraticreadout():
    return orc.readouts.QuadraticReadout(
        out_dim=3,
        res_dim=100,
        dtype=jnp.float64,
        seed=42,
    )


def test_single_quadraticreadout_dims(single_quadraticreadout):
    """Test that QuadraticReadout works with single reservoir (no chunks dimension)."""
    key = jax.random.key(123)
    out_dim = single_quadraticreadout.out_dim
    res_dim = single_quadraticreadout.res_dim

    # Test single state readout
    res_state = jax.random.normal(key, shape=(res_dim,))
    out_state = single_quadraticreadout.readout(res_state)

    assert out_state.shape == (out_dim,)
    assert jnp.all(jnp.isfinite(out_state))


def test_single_quadraticreadout_call(single_quadraticreadout):
    """Test QuadraticReadout __call__ method handles both single and batch inputs."""
    key = jax.random.key(456)
    out_dim = single_quadraticreadout.out_dim
    res_dim = single_quadraticreadout.res_dim

    # Test single input
    res_state = jax.random.normal(key, shape=(res_dim,))
    out_state = single_quadraticreadout(res_state)
    assert out_state.shape == (out_dim,)

    # Test batch input
    batch_res_state = jax.random.normal(key, shape=(5, res_dim))
    batch_out = single_quadraticreadout(batch_res_state)
    assert batch_out.shape == (5, out_dim)


def test_single_quadraticreadout_chunks_is_one(single_quadraticreadout):
    """Test that QuadraticReadout always has chunks=1."""
    assert single_quadraticreadout.chunks == 1


##################### ENSEMBLE LINEAR READOUT TESTS #####################


@pytest.mark.parametrize(
    "chunks,batch_size,out_dim",
    [
        (5, 32, 3),
        (3, 16, 4),
        (15, 17, 5),
    ],
)
def test_ensemble_readout_shapes(chunks, batch_size, out_dim):
    res_dim = 747
    readout = orc.readouts.EnsembleLinearReadout(out_dim, res_dim, chunks)

    inputs = jnp.ones((batch_size, chunks, res_dim))
    outputs = readout(inputs)
    assert outputs.shape == (
        batch_size,
        out_dim,
    )

    inputs = jnp.ones((chunks, res_dim))
    outputs = readout(inputs)
    assert outputs.shape == (out_dim,)
