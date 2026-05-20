import diffrax
import equinox as eqx
import jax
import jax.extend
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import orc
import orc.data


@pytest.fixture
def dummy_problem_params():
    """Set up dummy data for testing parallel ESNs."""
    Nx = 32
    time_steps = 1000
    frequencies = jnp.linspace(0.5, 5.0, Nx)
    time = jnp.arange(time_steps).reshape(-1, 1)
    dummy_data = jnp.sin(2 * jnp.pi * frequencies * time / time_steps)
    U_test = dummy_data[-100:]
    U_train = dummy_data[:-100]
    return Nx, U_train, U_test


####################### ESN TESTS #####################
def test_esn_train():
    """
    Test forecast on Lorenz system. Passes if forecast is accurate for 100 steps.
    """
    res_dim = 1000
    tN = 100
    dt = 0.01
    u0 = np.array([0.05, 1, 1.05])
    test_perc = 0.2
    fcast_len = 100

    # get data
    U, _ = orc.data.lorenz63(tN=tN, dt=dt, u0=u0)
    split_idx = int((1 - test_perc) * U.shape[0])
    U_train = U[:split_idx, :]
    U_test = U[split_idx:, :]

    # train esn
    esn = orc.forecaster.ESNForecaster(
        data_dim=3, res_dim=res_dim, seed=0, quadratic=True
    )
    esn, R = orc.forecaster.train_ESNForecaster(esn, U_train)

    # forecast
    U_pred = esn.forecast(fcast_len=fcast_len, res_state=R[-1])

    assert (jnp.linalg.norm(U_pred - U_test[:fcast_len, :]) / fcast_len) < 1e-3


def test_periodic_par_esn(dummy_problem_params):
    """Test periodic parallel ESN on dummy problem."""

    # test params
    res_dim = 200
    chunks = 16
    locality = 3
    fcast_len = 25

    # grab dummy data
    Nx, U_train, U_test = dummy_problem_params

    esn = orc.forecaster.ESNForecaster(
        data_dim=Nx,
        res_dim=res_dim,
        seed=0,
        chunks=chunks,
        locality=locality,
        periodic=True,  # Set periodic to True
    )

    # train esn
    esn, R = orc.forecaster.train_ESNForecaster(
        esn,
        U_train,
        initial_res_state=jax.numpy.zeros((chunks, res_dim), dtype=jnp.float64),
    )

    # forecast
    U_pred = esn.forecast(fcast_len=fcast_len, res_state=R[-1])
    assert (jnp.linalg.norm(U_pred - U_test[:fcast_len, :]) / fcast_len) < 1e-2


def test_nonperiodic_par_esn(dummy_problem_params):
    """Test nonperiodic parallel ESN on dummy problem."""
    # test params
    res_dim = 300
    chunks = 32
    locality = 2
    fcast_len = 25

    # grab dummy data
    Nx, U_train, U_test = dummy_problem_params

    # init esn
    esn = orc.forecaster.ESNForecaster(
        data_dim=Nx,
        res_dim=res_dim,
        seed=0,
        chunks=chunks,
        locality=locality,
        periodic=False,  # Set periodic to True
    )

    # train esn
    esn, R = orc.forecaster.train_ESNForecaster(
        esn,
        U_train,
        initial_res_state=jax.numpy.zeros((chunks, res_dim), dtype=jnp.float64),
    )

    # forecast
    U_pred = esn.forecast(fcast_len=fcast_len, res_state=R[-1])
    assert (jnp.linalg.norm(U_pred - U_test[:fcast_len, :]) / fcast_len) < 1e-2


def test_forecast_from_IC(dummy_problem_params):
    """Test forecast from IC vs forecast from reservoir state."""
    res_dim = 100
    chunks = 32
    locality = 2
    fcast_len = 25

    # grab dummy data
    Nx, U_train, U_test = dummy_problem_params

    esn = orc.forecaster.ESNForecaster(
        data_dim=Nx,
        res_dim=res_dim,
        seed=0,
        chunks=chunks,
        locality=locality,
        periodic=False,
    )

    esn, R = orc.forecaster.train_ESNForecaster(
        esn,
        U_train,
        initial_res_state=jax.numpy.zeros((chunks, res_dim), dtype=jnp.float64),
    )
    U_pred1 = esn.forecast(fcast_len=fcast_len, res_state=R[-1])
    U_pred2 = esn.forecast_from_IC(fcast_len, U_train[-101:])
    assert jnp.allclose(U_pred1, U_pred2)


###################### CESN TESTS #####################


def test_cesn_train():
    """
    Test forecast on Lorenz system with continuous ESN. Passes if forecast is accurate.
    """
    res_dim = 200
    tN = 100
    dt = 0.02
    u0 = np.array([0.05, 1, 1.05])
    test_perc = 0.2
    fcast_len = 100

    # get data
    U, ts = orc.data.lorenz63(tN=tN, dt=dt, u0=u0)
    split_idx = int((1 - test_perc) * U.shape[0])
    U_train = U[:split_idx, :]
    U_test = U[split_idx:, :]
    ts_train = ts[:split_idx]
    ts_test = jnp.arange(0, fcast_len, dtype=jnp.float64) * dt

    # train cesn
    solver = diffrax.Euler()
    stepsize_controller = diffrax.ConstantStepSize()  # faster for testing
    cesn = orc.forecaster.CESNForecaster(
        data_dim=3,
        res_dim=res_dim,
        seed=0,
        stepsize_controller=stepsize_controller,
        solver=solver,
    )
    cesn, R = orc.forecaster.train_CESNForecaster(cesn, U_train, ts_train)

    # forecast
    U_pred = cesn.forecast(ts=ts_test, res_state=R[-1])

    assert (jnp.linalg.norm(U_pred - U_test[:fcast_len, :]) / fcast_len) < 1e-2


def test_periodic_par_cesn(dummy_problem_params):
    """Test periodic parallel CESN on dummy problem."""
    # test params
    res_dim = 300
    chunks = 16
    locality = 3
    fcast_len = 10

    # grab dummy data
    Nx, U_train, U_test = dummy_problem_params
    ts_train = jnp.linspace(0, 10, U_train.shape[0])
    dt = ts_train[1] - ts_train[0]
    ts_test = jnp.arange(0, fcast_len, dtype=jnp.float64) * dt

    # init cesn
    cesn = orc.forecaster.CESNForecaster(
        data_dim=Nx,
        res_dim=res_dim,
        seed=0,
        chunks=chunks,
        locality=locality,
        periodic=True,
        time_const=25.0,
    )

    # train cesn
    cesn, R = orc.forecaster.train_CESNForecaster(
        cesn,
        U_train,
        ts_train,
        initial_res_state=jax.numpy.zeros((chunks, res_dim), dtype=jnp.float64),
    )

    # forecast
    U_pred = cesn.forecast(ts=ts_test, res_state=R[-1])
    assert (jnp.linalg.norm(U_pred - U_test[:fcast_len, :]) / fcast_len) < 1e-2


def test_nonperiodic_par_cesn(dummy_problem_params):
    """Test nonperiodic parallel CESN on dummy problem."""
    # test params
    res_dim = 200
    chunks = 32
    locality = 2
    fcast_len = 25

    # grab dummy data
    Nx, U_train, U_test = dummy_problem_params
    ts_train = jnp.linspace(0, 10, U_train.shape[0])
    dt = ts_train[1] - ts_train[0]
    ts_test = jnp.arange(0, fcast_len) * dt  # Time values for testing

    # init cesn
    cesn = orc.forecaster.CESNForecaster(
        data_dim=Nx,
        res_dim=res_dim,
        seed=0,
        chunks=chunks,
        locality=locality,
        periodic=False,
        time_const=25.0,
    )

    # train cesn
    cesn, R = orc.forecaster.train_CESNForecaster(
        cesn,
        U_train,
        ts_train,
        initial_res_state=jax.numpy.zeros((chunks, res_dim), dtype=jnp.float64),
    )

    # forecast
    U_pred = cesn.forecast(ts=ts_test, res_state=R[-1])
    assert (jnp.linalg.norm(U_pred - U_test[:fcast_len, :]) / fcast_len) < 1e-2


def test_forecast_from_IC_CESN(dummy_problem_params):
    """Test forecast from IC vs forecast from reservoir state."""
    res_dim = 300
    chunks = 8
    locality = 3
    fcast_len = 10

    # grab dummy data
    Nx, U_train, U_test = dummy_problem_params
    ts_train = jnp.linspace(0, 10, U_train.shape[0])
    dt = ts_train[1] - ts_train[0]
    ts_test = (
        jnp.arange(0, fcast_len, dtype=jnp.float64) * dt
    )  # Time values for testing

    esn = orc.forecaster.CESNForecaster(
        data_dim=Nx,
        res_dim=res_dim,
        seed=0,
        chunks=chunks,
        locality=locality,
        periodic=False,
        time_const=20.0,
        quadratic=True,
    )

    esn, R = orc.forecaster.train_CESNForecaster(
        esn,
        U_train,
        ts_train,
        initial_res_state=jax.numpy.zeros((chunks, res_dim), dtype=jnp.float64),
    )
    U_pred1 = esn.forecast(ts=ts_test, res_state=R[-1])
    U_pred2 = esn.forecast_from_IC(ts=ts_test, spinup_data=U_train)
    max_diff = jnp.max(jnp.abs(U_pred1 - U_pred2))

    # Very large tolerance because forcing is non-deterministic on GPU
    tol = 1e-1 if jax.extend.backend.get_backend().platform == "gpu" else 1e-12

    assert jnp.allclose(U_pred1, U_pred2, atol=tol), (
        f"Forecast from IC produced different values, max diff: {max_diff}"
    )


@pytest.mark.parametrize(
    "solver, controller",
    [
        (diffrax.Euler(), diffrax.ConstantStepSize()),
        (diffrax.Tsit5(), diffrax.PIDController(rtol=1e-3, atol=1e-6)),
        (diffrax.Dopri8(), diffrax.PIDController(rtol=1e-4, atol=1e-7)),
    ],
)
def test_cesn_different_solvers(solver, controller):
    """Test CESN with different diffrax solvers and controllers."""
    res_dim = 200
    data_dim = 3
    tN = 20
    dt = 0.02
    fcast_len = 10

    # Generate Lorenz data for testing
    u0 = jnp.array([0.5, 0.5, 0.5])
    U, ts = orc.data.lorenz63(tN=tN, dt=dt, u0=u0)

    # Split into train/test
    split_idx = int(0.8 * U.shape[0])
    U_train = U[:split_idx, :]
    ts_train = ts[:split_idx]
    ts_test = jnp.arange(0, fcast_len) * dt

    cesn = orc.forecaster.CESNForecaster(
        data_dim=data_dim,
        res_dim=res_dim,
        seed=0,
        solver=solver,
        stepsize_controller=controller,
    )

    # Train the model
    cesn, R = orc.forecaster.train_CESNForecaster(cesn, U_train, ts_train, beta=1e-6)

    # Test forecasting ability
    U_pred = cesn.forecast(ts=ts_test, res_state=R[-1])

    # Ensure output is well-formed (finitely valued, correct shape)
    assert U_pred.shape == (fcast_len, data_dim)
    assert jnp.all(jnp.isfinite(U_pred))


def test_esn_batched_vmap_equivalence(dummy_problem_params):
    """Test that batched vmap produces identical results to non-batched vmap."""
    Nx, U_train, U_test = dummy_problem_params

    # Use parallel ESN with multiple chunks to test batching
    chunks = 8
    res_dim = 500

    # Create ESN model
    esn = orc.forecaster.ESNForecaster(
        data_dim=Nx, res_dim=res_dim, chunks=chunks, seed=42
    )

    # Train without batching (default)
    esn_unbatched, R_unbatched = orc.forecaster.train_ESNForecaster(
        esn, U_train[:100], batch_size=None
    )

    # Train with batching
    esn_batched, R_batched = orc.forecaster.train_ESNForecaster(
        esn, U_train[:100], batch_size=4
    )

    # Large tolerance because forcing is non-deterministic on GPU
    tol = 1e-5 if jax.extend.backend.get_backend().platform == "gpu" else 1e-12

    # Results should be identical
    assert jnp.allclose(esn_unbatched.readout.wout, esn_batched.readout.wout, atol=tol)
    assert jnp.allclose(R_unbatched, R_batched, atol=tol)


def test_cesn_batched_vmap_equivalence(dummy_problem_params):
    """Test that batched vmap produces same results as non-batched vmap for CESN."""
    Nx, U_train, U_test = dummy_problem_params

    chunks = 8
    res_dim = 500
    dt = 0.01
    t_train = jnp.arange(100) * dt

    cesn = orc.forecaster.CESNForecaster(
        data_dim=Nx, res_dim=res_dim, chunks=chunks, seed=42
    )

    cesn_unbatched, R_unbatched = orc.forecaster.train_CESNForecaster(
        cesn, U_train[:100], t_train, batch_size=None
    )

    cesn_batched, R_batched = orc.forecaster.train_CESNForecaster(
        cesn, U_train[:100], t_train, batch_size=3
    )

    # Large tolerance because forcing is non-deterministic on GPU
    tol = 1e-5 if jax.extend.backend.get_backend().platform == "gpu" else 1e-12

    # Results should be identical
    assert jnp.allclose(
        cesn_unbatched.readout.wout, cesn_batched.readout.wout, atol=tol
    )
    assert jnp.allclose(R_unbatched, R_batched, atol=tol)


def test_batched_vmap_different_batch_sizes():
    """Test that different batch sizes produce identical results."""
    Nx = 32
    chunks = 4
    res_dim = 1000

    time_steps = 2000
    frequencies = jnp.linspace(0.5, 5.0, Nx)
    time = jnp.arange(time_steps).reshape(-1, 1)
    dummy_data = jnp.sin(2 * jnp.pi * frequencies * time / time_steps)

    esn = orc.forecaster.ESNForecaster(
        data_dim=Nx, res_dim=res_dim, chunks=chunks, seed=123
    )

    trained_models = []
    reservoir_states = []
    batch_sizes = [None, 1, 2, 4, 6]

    for batch_size in batch_sizes:
        esn_trained, R = orc.forecaster.train_ESNForecaster(
            esn, dummy_data, batch_size=batch_size
        )
        trained_models.append(esn_trained)
        reservoir_states.append(R)

    output_0 = trained_models[0].readout(reservoir_states[0])
    for i in range(1, len(trained_models)):
        output_i = trained_models[i].readout(reservoir_states[i])

        max_diff = jnp.max(jnp.abs(output_0 - output_i))
        assert jnp.allclose(output_0, output_i, atol=1e-8), (
            f"Batch size {batch_sizes[i]} produced different reconstruction",
            f"max diff: {max_diff}",
        )


####################### ESN TESTS #####################
def test_ensemble_esn_train():
    """
    Test forecast on Lorenz system. Passes if forecast is accurate for 100 steps.
    """
    res_dim = 1000
    chunks = 6
    tN = 100
    dt = 0.01
    u0 = np.array([0.05, 1, 1.05])
    test_perc = 0.2
    fcast_len = 100

    # get data
    U, _ = orc.data.lorenz63(tN=tN, dt=dt, u0=u0)
    split_idx = int((1 - test_perc) * U.shape[0])
    U_train = U[:split_idx, :]
    U_test = U[split_idx:, :]

    # train esn
    esn = orc.forecaster.EnsembleESNForecaster(
        data_dim=3, res_dim=res_dim, seed=0, chunks=chunks
    )
    esn, R = orc.forecaster.train_EnsembleESNForecaster(esn, U_train)

    # forecast
    U_pred = esn.forecast(fcast_len=fcast_len, res_state=R[-1])

    assert (jnp.linalg.norm(U_pred - U_test[:fcast_len, :]) / fcast_len) < 1e-3


def test_ensemble_forecast_from_IC(dummy_problem_params):
    """Test forecast from IC vs forecast from reservoir state."""
    res_dim = 100
    chunks = 32
    locality = 2
    fcast_len = 25

    # grab dummy data
    Nx, U_train, U_test = dummy_problem_params

    esn = orc.forecaster.ESNForecaster(
        data_dim=Nx,
        res_dim=res_dim,
        seed=0,
        chunks=chunks,
        locality=locality,
        periodic=False,
    )

    esn, R = orc.forecaster.train_ESNForecaster(
        esn,
        U_train,
        initial_res_state=jax.numpy.zeros((chunks, res_dim), dtype=jnp.float64),
    )
    U_pred1 = esn.forecast(fcast_len=fcast_len, res_state=R[-1])
    U_pred2 = esn.forecast_from_IC(fcast_len, U_train[-101:])
    assert jnp.allclose(U_pred1, U_pred2)


####################### UNIFIED train_RCForecaster TESTS #####################


def test_train_rcforecaster_esn():
    """Test train_RCForecaster produces same result as train_ESNForecaster."""
    res_dim = 500
    tN = 50
    dt = 0.01
    u0 = np.array([0.05, 1, 1.05])

    U, _ = orc.data.lorenz63(tN=tN, dt=dt, u0=u0)
    U_train = U[:4000, :]

    esn = orc.forecaster.ESNForecaster(
        data_dim=3, res_dim=res_dim, seed=0, quadratic=True
    )

    esn_old, R_old = orc.forecaster.train_ESNForecaster(esn, U_train)
    esn_new, R_new = orc.forecaster.train_RCForecaster(esn, U_train)

    assert jnp.allclose(esn_old.readout.wout, esn_new.readout.wout)
    assert jnp.allclose(R_old, R_new)


def test_train_rcforecaster_cesn():
    """Test train_RCForecaster with ts= produces same result as train_CESNForecaster."""
    res_dim = 300
    tN = 30
    dt = 0.01
    u0 = np.array([0.05, 1, 1.05])

    U, ts = orc.data.lorenz63(tN=tN, dt=dt, u0=u0)
    U_train = U[:2000, :]
    ts_train = ts[:2000]

    cesn = orc.forecaster.CESNForecaster(
        data_dim=3, res_dim=res_dim, seed=0
    )

    cesn_old, R_old = orc.forecaster.train_CESNForecaster(cesn, U_train, ts_train)
    cesn_new, R_new = orc.forecaster.train_RCForecaster(cesn, U_train, ts=ts_train)

    assert jnp.allclose(cesn_old.readout.wout, cesn_new.readout.wout)
    assert jnp.allclose(R_old, R_new)


def test_train_rcforecaster_ensemble():
    """Test train_RCForecaster produces same result as train_EnsembleESNForecaster."""
    res_dim = 500
    chunks = 6
    tN = 50
    dt = 0.01
    u0 = np.array([0.05, 1, 1.05])

    U, _ = orc.data.lorenz63(tN=tN, dt=dt, u0=u0)
    U_train = U[:4000, :]

    esn = orc.forecaster.EnsembleESNForecaster(
        data_dim=3, res_dim=res_dim, seed=0, chunks=chunks
    )

    esn_old, R_old = orc.forecaster.train_EnsembleESNForecaster(esn, U_train)
    esn_new, R_new = orc.forecaster.train_RCForecaster(esn, U_train)

    assert jnp.allclose(esn_old.readout.wout, esn_new.readout.wout)
    assert jnp.allclose(R_old, R_new)


def test_train_rcforecaster_custom_readout():
    """Test train_RCForecaster with a custom chunks=0 readout."""

    class SimpleReadout(orc.readouts.ReadoutBase):
        wout: jnp.ndarray

        def __init__(self, out_dim, res_dim):
            super().__init__(out_dim, res_dim)
            self.wout = jnp.zeros((out_dim, res_dim))

        def readout(self, res_state):
            return self.wout @ res_state

    class SimpleForecaster(orc.forecaster.RCForecasterBase):
        pass

    data_dim = 3
    res_dim = 200
    embedding = orc.embeddings.LinearEmbedding(
        in_dim=data_dim, res_dim=res_dim, scaling=1.0, seed=0
    )
    driver = orc.drivers.ESNDriver(res_dim=res_dim, seed=0)
    readout = SimpleReadout(out_dim=data_dim, res_dim=res_dim)

    model = SimpleForecaster(driver, readout, embedding)
    assert model.chunks == 0

    tN = 50
    dt = 0.01
    u0 = np.array([0.05, 1, 1.05])
    U, _ = orc.data.lorenz63(tN=tN, dt=dt, u0=u0)
    U_train = U[:4000, :]

    trained_model, tot_res_seq = orc.forecaster.train_RCForecaster(model, U_train)

    # Verify wout was updated from zeros
    assert not jnp.allclose(trained_model.readout.wout, jnp.zeros((data_dim, res_dim)))
    # Verify shapes
    assert trained_model.readout.wout.shape == (data_dim, res_dim)
    assert tot_res_seq.shape[1] == res_dim


##################### JAX TRANSFORM & AD TESTS #####################


@pytest.fixture
def esn_forecaster():
    return orc.forecaster.ESNForecaster(data_dim=3, res_dim=200, seed=0)


@pytest.fixture
def ensemble_esn_forecaster():
    return orc.forecaster.EnsembleESNForecaster(
        data_dim=3, res_dim=200, chunks=4, seed=0
    )


@pytest.fixture
def cesn_forecaster():
    # Fix max_steps and stepsize_controller with diff'ble settings
    return orc.forecaster.CESNForecaster(
        data_dim=3,
        res_dim=200,
        seed=0,
        solver=diffrax.Euler(),
        stepsize_controller=diffrax.ConstantStepSize(),
        max_steps=4096,
    )


@pytest.mark.parametrize(
    "forecaster_fixture", ["esn_forecaster", "ensemble_esn_forecaster"]
)
def test_forecaster_transform_stability(forecaster_fixture, request):
    """Verify discrete forecasters are compatible with vmap and jit."""
    model = request.getfixturevalue(forecaster_fixture)
    key = jax.random.key(999)
    fcast_len = 5
    batch_size = 3

    state_shape = model.driver.default_state().shape
    batch_state = jax.random.normal(key, shape=(batch_size, *state_shape))

    def fwd(res_state):
        return model.forecast(fcast_len, res_state)

    vmap_fwd = eqx.filter_vmap(fwd)
    assert jnp.allclose(
        vmap_fwd(batch_state),
        jnp.stack([fwd(s) for s in batch_state]),
    )

    jit_fwd = eqx.filter_jit(fwd)
    assert jnp.all(jnp.isfinite(jit_fwd(batch_state[0])))


@pytest.mark.parametrize(
    "forecaster_fixture", ["esn_forecaster", "ensemble_esn_forecaster"]
)
def test_forecaster_differentiability(forecaster_fixture, request):
    """Verify gradients flow backward through forecast via Equinox."""
    model = request.getfixturevalue(forecaster_fixture)
    key = jax.random.key(999)
    fcast_len = 5

    res_state = jax.random.normal(key, shape=model.driver.default_state().shape)

    @eqx.filter_value_and_grad
    def loss_fn(m, s):
        return jnp.sum(m.forecast(fcast_len, s))

    loss, grads = loss_fn(model, res_state)
    assert jnp.isfinite(loss)

    def check_finite(g):
        if eqx.is_array(g) and jnp.issubdtype(g.dtype, jnp.inexact):
            assert jnp.all(jnp.isfinite(g))

    jax.tree_util.tree_map(check_finite, grads)


def test_cesn_forecast_transform_stability(cesn_forecaster):
    """Verify CESN continuous forecast is compatible with vmap and jit."""
    model = cesn_forecaster
    key = jax.random.key(999)
    batch_size = 3
    ts = jnp.arange(5, dtype=jnp.float64) * 0.02

    state_shape = model.driver.default_state().shape
    batch_state = jax.random.normal(key, shape=(batch_size, *state_shape))

    def fwd(res_state):
        return model.forecast(ts, res_state)

    vmap_fwd = eqx.filter_vmap(fwd)
    assert jnp.allclose(
        vmap_fwd(batch_state),
        jnp.stack([fwd(s) for s in batch_state]),
    )

    jit_fwd = eqx.filter_jit(fwd)
    assert jnp.all(jnp.isfinite(jit_fwd(batch_state[0])))


def test_cesn_forecast_differentiability(cesn_forecaster):
    """Verify gradients flow backward through CESN forecast via Equinox."""
    model = cesn_forecaster
    key = jax.random.key(999)
    ts = jnp.arange(5, dtype=jnp.float64) * 0.02

    res_state = jax.random.normal(key, shape=model.driver.default_state().shape)

    @eqx.filter_value_and_grad
    def loss_fn(m, s):
        return jnp.sum(m.forecast(ts, s))

    loss, grads = loss_fn(model, res_state)
    assert jnp.isfinite(loss)

    def check_finite(g):
        if eqx.is_array(g) and jnp.issubdtype(g.dtype, jnp.inexact):
            assert jnp.all(jnp.isfinite(g))
    jax.tree_util.tree_map(check_finite, grads)


##################### GRU FORECAST SMOKE TEST #####################


def test_gru_forecaster_smoke():
    """Built-in GRUDriver forecasts Rossler.
    """

    class GRUForecaster(orc.forecaster.RCForecasterBase):
        pass

    data_dim = 3
    res_dim = 300
    spinup = 100
    fcast_len = 50

    U, _ = orc.data.rossler(tN=40, dt=0.02, u0=jnp.array([-10.0, 2.0, 1.0]))
    split = int(U.shape[0] * 0.8)
    U_train, U_test = U[:split], U[split:]

    model = GRUForecaster(
        orc.drivers.GRUDriver(res_dim=res_dim, input_rescaling=5.0, seed=0),
        orc.readouts.LinearReadout(out_dim=data_dim, res_dim=res_dim, seed=0),
        orc.embeddings.LinearEmbedding(
            in_dim=data_dim, res_dim=res_dim, scaling=0.1, seed=0
        ),
    )
    model, _ = orc.forecaster.train_RCForecaster(
        model, train_seq=U_train, spinup=spinup, beta=1e-7
    )
    U_pred = model.forecast_from_IC(
        fcast_len=fcast_len, spinup_data=U_train[-spinup:]
    )

    assert U_pred.shape == (fcast_len, data_dim)
    assert jnp.all(jnp.isfinite(U_pred))
    assert jnp.max(jnp.abs(U_pred)) < 100
    nrmse = jnp.linalg.norm(
        (U_pred - U_test[:fcast_len]) / jnp.std(U_train, axis=0), axis=1
    ) / jnp.sqrt(data_dim)
    assert jnp.mean(nrmse) < 0.5
