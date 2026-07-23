import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

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


@pytest.mark.parametrize("chunks", [1, 4])
def test_parallel_linear_prepare_target(chunks):
    """
    Test that prepare_target reshapes (seq_len, out_dim) to
    (seq_len, chunks, out_dim/chunks).
    """
    out_dim = 12
    res_dim = 100
    readout = orc.readouts.ParallelLinearReadout(
        out_dim=out_dim, res_dim=res_dim, chunks=chunks
    )
    target_seq = jnp.ones((50, out_dim))
    result = readout.prepare_target(target_seq)
    assert result.shape == (50, chunks, out_dim // chunks)


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


@pytest.mark.parametrize("chunks", [1, 4])
def test_parallel_nonlinear_prepare_train(chunks):
    """Test that ParallelNonlinearReadout.prepare_train applies nonlinear transform."""
    res_dim = 12
    out_dim = 12
    readout = orc.readouts.ParallelNonlinearReadout(
        out_dim=out_dim,
        res_dim=res_dim,
        nonlin_list=[lambda x: x**2],
        chunks=chunks,
        dtype=jnp.float64,
    )
    res_seq = jax.random.normal(jax.random.key(0), shape=(50, chunks, res_dim))
    result = readout.prepare_train(res_seq)
    assert result.shape == (50, chunks, res_dim)
    expected = readout.nonlinear_transform(res_seq)
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize("chunks", [1, 4])
def test_parallel_nonlinear_prepare_target(chunks):
    """Test that ParallelNonlinearReadout.prepare_target reshapes correctly."""
    out_dim = 12
    readout = orc.readouts.ParallelNonlinearReadout(
        out_dim=out_dim,
        res_dim=12,
        nonlin_list=[lambda x: x**2],
        chunks=chunks,
        dtype=jnp.float64,
    )
    target_seq = jnp.ones((50, out_dim))
    result = readout.prepare_target(target_seq)
    assert result.shape == (50, chunks, out_dim // chunks)


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


def test_single_linearreadout_prepare_train(single_linearreadout):
    """Test that LinearReadout.prepare_train unsqueezes to (seq_len, 1, res_dim)."""
    res_seq = jnp.ones((50, single_linearreadout.res_dim))
    result = single_linearreadout.prepare_train(res_seq)
    assert result.shape == (50, 1, single_linearreadout.res_dim)
    assert jnp.array_equal(result[:, 0, :], res_seq)


def test_single_linearreadout_prepare_target(single_linearreadout):
    """Test that LinearReadout.prepare_target unsqueezes to (seq_len, 1, out_dim)."""
    target_seq = jnp.ones((50, single_linearreadout.out_dim))
    result = single_linearreadout.prepare_target(target_seq)
    assert result.shape == (50, 1, single_linearreadout.out_dim)
    assert jnp.array_equal(result[:, 0, :], target_seq)


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


def test_single_nonlinearreadout_prepare_train(single_nonlinearreadout):
    """Test that NonlinearReadout.prepare_train applies transform and unsqueezes."""
    res_seq = jnp.ones((50, single_nonlinearreadout.res_dim))
    result = single_nonlinearreadout.prepare_train(res_seq)
    assert result.shape == (50, 1, single_nonlinearreadout.res_dim)
    # Verify nonlinear transform was applied (not just identity)
    expected = single_nonlinearreadout.nonlinear_transform(res_seq)
    assert jnp.allclose(result[:, 0, :], expected)


def test_single_nonlinearreadout_prepare_target(single_nonlinearreadout):
    """Test that NonlinearReadout.prepare_target unsqueezes to (seq_len, 1, out_dim)."""
    target_seq = jnp.ones((50, single_nonlinearreadout.out_dim))
    result = single_nonlinearreadout.prepare_target(target_seq)
    assert result.shape == (50, 1, single_nonlinearreadout.out_dim)
    assert jnp.array_equal(result[:, 0, :], target_seq)


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


def test_single_quadraticreadout_prepare_train(single_quadraticreadout):
    """Test that QuadraticReadout inherits prepare_train from NonlinearReadout."""
    res_seq = jnp.ones((50, single_quadraticreadout.res_dim))
    result = single_quadraticreadout.prepare_train(res_seq)
    assert result.shape == (50, 1, single_quadraticreadout.res_dim)
    expected = single_quadraticreadout.nonlinear_transform(res_seq)
    assert jnp.allclose(result[:, 0, :], expected)


def test_single_quadraticreadout_prepare_target(single_quadraticreadout):
    """Test that QuadraticReadout inherits prepare_target from NonlinearReadout."""
    target_seq = jnp.ones((50, single_quadraticreadout.out_dim))
    result = single_quadraticreadout.prepare_target(target_seq)
    assert result.shape == (50, 1, single_quadraticreadout.out_dim)


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


@pytest.mark.parametrize("chunks,out_dim", [(5, 3), (3, 4)])
def test_ensemble_prepare_target(chunks, out_dim):
    """Test that EnsembleLinearReadout.prepare_target repeats target across chunks."""
    res_dim = 100
    readout = orc.readouts.EnsembleLinearReadout(out_dim, res_dim, chunks)
    target_seq = jax.random.normal(jax.random.key(0), shape=(50, out_dim))
    result = readout.prepare_target(target_seq)
    assert result.shape == (50, chunks, out_dim)
    # Each chunk should get the same target
    for c in range(chunks):
        assert jnp.array_equal(result[:, c, :], target_seq)


##################### CUSTOM READOUT (chunks=0) TESTS #####################


class CustomReadout(orc.readouts.ReadoutBase):
    """Custom readout mimicking the notebook pattern, with wout convention."""

    wout: jnp.ndarray

    def __init__(self, out_dim, res_dim):
        super().__init__(out_dim, res_dim)
        self.wout = jnp.zeros((out_dim, res_dim))

    def readout(self, res_state):
        return self.wout @ res_state


def test_custom_readout_chunks_default():
    """Custom readouts inheriting from ReadoutBase should have chunks=0."""
    readout = CustomReadout(out_dim=3, res_dim=100)
    assert readout.chunks == 0


def test_custom_readout_prepare_train_passthrough():
    """prepare_train should be identity for chunks=0 readouts."""
    readout = CustomReadout(out_dim=3, res_dim=100)
    res_seq = jnp.ones((50, 100))
    result = readout.prepare_train(res_seq)
    assert result.shape == (50, 100)
    assert jnp.array_equal(result, res_seq)


def test_custom_readout_prepare_target_passthrough():
    """prepare_target should be identity for chunks=0 readouts."""
    readout = CustomReadout(out_dim=3, res_dim=100)
    target_seq = jnp.ones((50, 3))
    result = readout.prepare_target(target_seq)
    assert result.shape == (50, 3)
    assert jnp.array_equal(result, target_seq)


def test_custom_readout_set_wout():
    """set_wout should return a new readout with updated weights."""
    readout = CustomReadout(out_dim=3, res_dim=100)
    new_wout = jnp.ones((3, 100))
    new_readout = readout.set_wout(new_wout)
    assert jnp.array_equal(new_readout.wout, new_wout)
    assert jnp.array_equal(readout.wout, jnp.zeros((3, 100)))


def test_custom_readout_call():
    """Custom readout with chunks=0 should handle 1D and 2D inputs."""
    readout = CustomReadout(out_dim=3, res_dim=10)
    new_wout = jnp.eye(3, 10)
    readout = readout.set_wout(new_wout)

    # Single state
    res_state = jnp.arange(10, dtype=jnp.float64)
    out = readout(res_state)
    assert out.shape == (3,)

    # Batch of states
    batch = jnp.ones((5, 10))
    out = readout(batch)
    assert out.shape == (5, 3)


##################### JAX TRANSFORM & AD TESTS #####################

# Single-reservoir readouts accept a 1D (res_dim,) state; the parallel/ensemble
# variants accept a 2D (chunks, res_dim) state. LinearReadout/NonlinearReadout
# (and QuadraticReadout via NonlinearReadout) are the single-reservoir classes;
# the Parallel*/Ensemble* classes are their base classes, not instances of them.
_SINGLE_READOUTS = (orc.readouts.LinearReadout, orc.readouts.NonlinearReadout)


def _readout_state_shape(readout):
    """Per-reservoir-state shape accepted by ``readout.readout``."""
    if isinstance(readout, _SINGLE_READOUTS):
        return (readout.res_dim,)
    return (readout.chunks, readout.res_dim)


@pytest.mark.parametrize(
    "readout_fixture",
    [
        "linearreadout",
        "single_linearreadout",
        "single_nonlinearreadout",
        "single_quadraticreadout",
    ],
)
def test_readout_transform_stability(readout_fixture, request):
    """Verify readouts are compatible with vmap and jit."""
    readout = request.getfixturevalue(readout_fixture)
    key = jax.random.key(999)
    batch_size = 3

    shape = _readout_state_shape(readout)
    batch_state = jax.random.normal(key, shape=(batch_size, *shape))

    vmap_readout = eqx.filter_vmap(readout.readout)
    assert jnp.allclose(
        vmap_readout(batch_state),
        readout.batch_readout(batch_state),
    )

    jit_readout = eqx.filter_jit(readout.readout)
    assert jnp.all(jnp.isfinite(jit_readout(batch_state[0])))


@pytest.mark.parametrize(
    "readout_fixture",
    [
        "linearreadout",
        "single_linearreadout",
        "single_nonlinearreadout",
        "single_quadraticreadout",
    ],
)
def test_readout_differentiability(readout_fixture, request):
    """Verify gradients flow backward through the readout step via Equinox."""
    readout = request.getfixturevalue(readout_fixture)
    key = jax.random.key(999)

    res_state = jax.random.normal(key, shape=_readout_state_shape(readout))

    @eqx.filter_value_and_grad
    def loss_fn(model, s):
        return jnp.sum(model.readout(s))

    loss, grads = loss_fn(readout, res_state)
    assert jnp.isfinite(loss)

    def check_finite(g):
        if eqx.is_array(g) and jnp.issubdtype(g.dtype, jnp.inexact):
            assert jnp.all(jnp.isfinite(g))

    jax.tree_util.tree_map(check_finite, grads)
