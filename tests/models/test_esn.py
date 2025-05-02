import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
    esn = orc.models.ESN(data_dim=3, res_dim=res_dim, seed=0)
    esn, R = orc.models.esn.train_ESN_forecaster(esn, U_train)

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

    # grab KS data
    Nx = 64
    dummy_data = jnp.repeat(jnp.arange(Nx).reshape(1,-1), 1000, axis=0)
    key = jax.random.key(0)
    # some noise increases robustness of ESN forecast
    U_train = dummy_data + jax.random.normal(key=key, shape=(1000,Nx)) * 0.02
    U_test = dummy_data
    # init esn
    esn = orc.models.ESN(
        data_dim=Nx,
        res_dim=res_dim,
        seed=0,
        chunks=chunks,
        locality=locality,
        periodic=True,  # Set periodic to True
    )

    # train esn
    esn, R = orc.models.esn.train_ESN_forecaster(
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

    # grab KS data
    Nx = 128
    dummy_data = jnp.repeat(jnp.arange(Nx).reshape(1,-1), 2000, axis=0) * 2
    key = jax.random.key(0)
    # some noise increases robustness of ESN forecast
    U_train = dummy_data + jax.random.normal(key=key, shape=(2000, Nx)) * 0.02
    U_test = dummy_data
    # init esn
    esn = orc.models.ESN(
        data_dim=Nx,
        res_dim=res_dim,
        seed=0,
        chunks=chunks,
        locality=locality,
        periodic=False,  # Set periodic to True
    )

    # train esn
    esn, R = orc.models.esn.train_ESN_forecaster(
        esn,
        U_train,
        initial_res_state=jax.numpy.zeros((chunks, res_dim), dtype=jnp.float64),
    )

    # forecast
    U_pred = esn.forecast(fcast_len=fcast_len, res_state=R[-1])
    print(U_pred)
    assert (jnp.linalg.norm(U_pred - U_test[:fcast_len, :]) / fcast_len) < 1e-2

# TODO: separate the non periodic test below
    # train_len = 15000
    # input_sequence = jnp.array(U_train[-train_len - 1 : -1])
    # target_sequence = jnp.array(U_train[-train_len:])
    # NR = 1000
    # chunks = 4
    # locality = 6
    # esn = orc.models.ESN(
    #     data_dim=Nx, res_dim=NR, seed=2, chunks=chunks, locality=locality
    # )

    # esn, output_seq = orc.models.esn.train_ESN_forecaster(
    #     esn,
    #     input_sequence,
    #     target_sequence,
    #     spinup=200,
    #     initial_res_state=jax.numpy.zeros((chunks, NR), dtype=jnp.float64),
    #     beta=8e-8,
    # )

    # fcast = esn.forecast(U_test.shape[0], output_seq[-1])

    # assert jnp.linalg.norm(fcast[:50] - U_test[:50]) / (50 * Nx) < 1e-3

    # esn = orc.models.ESN(
    #     data_dim=Nx,
    #     res_dim=NR,
    #     seed=2,
    #     chunks=chunks,
    #     locality=locality,
    #     periodic=False,
    # )

    # esn, output_seq = orc.models.esn.train_ESN_forecaster(
    #     esn,
    #     input_sequence,
    #     target_sequence,
    #     spinup=200,
    #     initial_res_state=jax.numpy.zeros((chunks, NR), dtype=jnp.float64),
    #     beta=8e-8,
    # )

    # fcast = esn.forecast(U_test.shape[0], output_seq[-1])

    # assert jnp.linalg.norm(fcast[:50] - U_test[:50]) / (50 * Nx) < 1e-3


# @pytest.fixture
# def gen_KS_data():
#     tN = 1000
#     Nx = 64
#     U, t = orc.data.KS_1D(tN = tN, Nx = Nx) # use default parameters for KS_1D

#     # train-test split
#     test_perc = 0.2
#     split_idx = int((1 - test_perc) * U.shape[0])
#     U_train = U[:split_idx, :]
#     U_test = U[split_idx:, :]
#     return Nx, U_train, U_test
