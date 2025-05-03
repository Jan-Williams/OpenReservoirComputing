import jax
import jax.numpy as jnp
import numpy as np

import orc
import orc.data


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
    """Test periodic ESN on KS. Passes if forecast is accurate for 25 steps."""

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
