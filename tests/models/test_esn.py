import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import orc
import orc.data


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
    esn = orc.models.ESNForecaster(data_dim=3, res_dim=res_dim, seed=0)
    esn, R = orc.models.esn.train_ESNForecaster(esn, U_train)

    # forecast
    U_pred = esn.forecast(fcast_len=fcast_len, res_state=R[-1])

    assert (jnp.linalg.norm(U_pred - U_test[:fcast_len, :]) / fcast_len) < 1e-3


def test_periodic_par_esn():
    """Test periodic parallel ESN on dummy problem."""

    # test params
    res_dim = 200
    chunks = 16
    locality = 3
    fcast_len = 25

    # grab dummy data
    Nx = 64
    dummy_data = jnp.repeat(jnp.arange(Nx).reshape(1,-1), 1000, axis=0)
    key = jax.random.key(0)
    # some noise increases robustness of ESN forecast
    U_train = dummy_data + jax.random.normal(key=key, shape=(1000,Nx)) * 0.02
    U_test = dummy_data
    # init esn
    esn = orc.models.ESNForecaster(
        data_dim=Nx,
        res_dim=res_dim,
        seed=0,
        chunks=chunks,
        locality=locality,
        periodic=True,  # Set periodic to True
    )

    # train esn
    esn, R = orc.models.esn.train_ESNForecaster(
        esn,
        U_train,
        initial_res_state=jax.numpy.zeros((chunks, res_dim), dtype=jnp.float64),
    )

    # forecast
    U_pred = esn.forecast(fcast_len=fcast_len, res_state=R[-1])
    assert (jnp.linalg.norm(U_pred - U_test[:fcast_len, :]) / fcast_len) < 1e-2

def test_nonperiodic_par_esn():
    """Test nonperiodic parallel ESN on dummy problem. """
    # test params
    res_dim = 300
    chunks = 32
    locality = 2
    fcast_len = 25

    # create dummy data
    Nx = 128
    dummy_data = jnp.repeat(jnp.arange(Nx).reshape(1,-1), 2000, axis=0) * 2
    key = jax.random.key(0)
    # some noise increases robustness of ESN forecast
    U_train = dummy_data + jax.random.normal(key=key, shape=(2000, Nx)) * 0.02
    U_test = dummy_data

    # init esn
    esn = orc.models.ESNForecaster(
        data_dim=Nx,
        res_dim=res_dim,
        seed=0,
        chunks=chunks,
        locality=locality,
        periodic=False,  # Set periodic to True
    )

    # train esn
    esn, R = orc.models.esn.train_ESNForecaster(
        esn,
        U_train,
        initial_res_state=jax.numpy.zeros((chunks, res_dim), dtype=jnp.float64),
    )

    # forecast
    U_pred = esn.forecast(fcast_len=fcast_len, res_state=R[-1])
    assert (jnp.linalg.norm(U_pred - U_test[:fcast_len, :]) / fcast_len) < 1e-2



###################### CESN TESTS #####################

def test_cesn_train():
    """
    Test forecast on Lorenz system with continuous ESN. Passes if forecast is accurate.
    """
    res_dim = 400
    tN = 100
    dt = 0.01
    u0 = np.array([0.05, 1, 1.05])
    test_perc = 0.2
    fcast_len = 100

    # get data
    U, ts = orc.data.lorenz63(tN=tN, dt=dt, u0=u0)
    split_idx = int((1 - test_perc) * U.shape[0])
    U_train = U[:split_idx, :]
    U_test = U[split_idx:, :]
    ts_train = ts[:split_idx]
    ts_test = jnp.arange(0, fcast_len) * dt  # Time values for testing

    # train cesn
    cesn = orc.models.CESNForecaster(data_dim=3, res_dim=res_dim, seed=0)
    cesn, R = orc.models.esn.train_CESNForecaster(cesn, U_train, ts_train)

    # forecast
    U_pred = cesn.forecast(ts=ts_test, res_state=R[-1])

    assert (jnp.linalg.norm(U_pred - U_test[:fcast_len, :]) / fcast_len) < 1e-3


def test_periodic_par_cesn():
    """Test periodic parallel CESN on dummy problem."""
    # test params
    res_dim = 200
    chunks = 16
    locality = 3
    fcast_len = 25

    # grab dummy data
    Nx = 64
    dummy_data = jnp.repeat(jnp.sin(Nx).reshape(1,-1), 1000, axis=0)
    key = jax.random.key(0)
    # some noise increases robustness of ESN forecast
    U_train = dummy_data + jax.random.normal(key=key, shape=(1000,Nx)) * 0.02
    U_test = dummy_data
    ts_train = jnp.linspace(0, 10, 1000)  # Time values for training
    dt = ts_train[1] - ts_train[0]
    ts_test = jnp.arange(0, fcast_len) * dt  # Time values for testing

    # init cesn
    cesn = orc.models.CESNForecaster(
        data_dim=Nx,
        res_dim=res_dim,
        seed=0,
        chunks=chunks,
        locality=locality,
        periodic=True,
        time_const = 25.0,
    )

    # train cesn
    cesn, R = orc.models.esn.train_CESNForecaster(
        cesn,
        U_train,
        ts_train,
        initial_res_state=jax.numpy.zeros((chunks, res_dim), dtype=jnp.float64),
    )

    # forecast
    U_pred = cesn.forecast(ts=ts_test, res_state=R[-1])
    assert (jnp.linalg.norm(U_pred - U_test[:fcast_len, :]) / fcast_len) < 1e-2


def test_nonperiodic_par_cesn():
    """Test nonperiodic parallel CESN on dummy problem."""
    # test params
    res_dim = 200
    chunks = 32
    locality = 2
    fcast_len = 25

    # create dummy data
    Nx = 128
    dummy_data = jnp.repeat(jnp.sin(Nx).reshape(1,-1), 2000, axis=0) * 2
    key = jax.random.key(0)
    # some noise increases robustness of ESN forecast
    U_train = dummy_data + jax.random.normal(key=key, shape=(2000, Nx)) * 0.02
    U_test = dummy_data
    ts_train = jnp.linspace(0, 20, 2000)  # Time values for training
    dt = ts_train[1] - ts_train[0]
    ts_test = jnp.arange(0, fcast_len) * dt  # Time values for testing

    # init cesn
    cesn = orc.models.CESNForecaster(
        data_dim=Nx,
        res_dim=res_dim,
        seed=0,
        chunks=chunks,
        locality=locality,
        periodic=False,
        time_const = 25.0,
    )

    # train cesn
    cesn, R = orc.models.esn.train_CESNForecaster(
        cesn,
        U_train,
        ts_train,
        initial_res_state=jax.numpy.zeros((chunks, res_dim), dtype=jnp.float64),
    )

    # forecast
    U_pred = cesn.forecast(ts=ts_test, res_state=R[-1])
    assert (jnp.linalg.norm(U_pred - U_test[:fcast_len, :]) / fcast_len) < 1e-2

@pytest.mark.parametrize(
    "solver, controller",
    [
        (diffrax.Euler(), diffrax.ConstantStepSize()),
        (diffrax.Tsit5(), diffrax.PIDController(rtol=1e-3, atol=1e-6)),
        (diffrax.Dopri8(), diffrax.PIDController(rtol=1e-4, atol=1e-7)),
    ])
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

    cesn = orc.models.CESNForecaster(
        data_dim=data_dim,
        res_dim=res_dim,
        seed=0,
        solver=solver,
        stepsize_controller=controller
    )

    # Train the model
    cesn, R = orc.models.esn.train_CESNForecaster(
        cesn,
        U_train,
        ts_train,
        beta=1e-6
    )

    # Test forecasting ability
    U_pred = cesn.forecast(ts=ts_test, res_state=R[-1])

    # Ensure output is well-formed (finitely valued, correct shape)
    assert U_pred.shape == (fcast_len, data_dim)
    assert jnp.all(jnp.isfinite(U_pred))
