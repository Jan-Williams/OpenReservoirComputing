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
    esn = orc.models.ESNForecaster(data_dim=3, res_dim=res_dim, seed=0)
    esn, R = orc.models.esn.train_ESNForecaster(esn, U_train)

    # forecast
    U_pred = esn.forecast(fcast_len=fcast_len, res_state=R[-1])

    assert (jnp.linalg.norm(U_pred - U_test[:fcast_len, :]) / fcast_len) < 1e-3


def test_periodic_par_esn(gen_KS_data):
    #TODO: clean this test up a bit
    Nx = gen_KS_data[0]
    U_train = gen_KS_data[1]
    U_test = gen_KS_data[2]

    train_len = 15000
    input_sequence = jnp.array(U_train[-train_len - 1 : -1])
    target_sequence = jnp.array(U_train[-train_len:])
    NR = 1000
    chunks = 32
    locality = 3
    esn = orc.models.ESNForecaster(
        data_dim=Nx, res_dim=NR, seed=2, chunks=chunks, locality=locality
    )

    esn, output_seq = orc.models.esn.train_ESNForecaster(
        esn,
        input_sequence,
        target_sequence,
        spinup=200,
        initial_res_state=jax.numpy.zeros((chunks, NR), dtype=jnp.float64),
        beta=8e-8,
    )

    fcast = esn.forecast(U_test.shape[0], output_seq[-1])

    assert jnp.linalg.norm(fcast[:50] - U_test[:50]) / (50 * Nx) < 1e-3

    esn = orc.models.ESNForecaster(
        data_dim=Nx,
        res_dim=NR,
        seed=2,
        chunks=chunks,
        locality=locality,
        periodic=False,
    )

    esn, output_seq = orc.models.esn.train_ESNForecaster(
        esn,
        input_sequence,
        target_sequence,
        spinup=200,
        initial_res_state=jax.numpy.zeros((chunks, NR), dtype=jnp.float64),
        beta=8e-8,
    )

    fcast = esn.forecast(U_test.shape[0], output_seq[-1])

    assert jnp.linalg.norm(fcast[:50] - U_test[:50]) / (50 * Nx) < 1e-3


@pytest.fixture
def gen_KS_data():
    domain = (0, 32 * np.pi)  # x in [0, 32*pi]
    Nx = 256  # number of spatial points
    x0 = np.linspace(
        domain[0], domain[1], Nx, endpoint=True
    )  # Ensure periodic boundary condition of IC
    u0 = np.sin((16 / domain[1]) * np.pi * x0)  # initial condition
    TN = 1500  # time step
    dt = 0.05
    # U, t = KS_1D_PBC(u0, TN, dt, domain, Nx)  # Solve!
    U, t = orc.data.KS_1D(tN=TN,u0=u0,dt=dt,domain=domain,Nx=Nx)  # Solve!
    U = U.T

    # train-test split
    test_perc = 0.2
    split_idx = int((1 - test_perc) * U.shape[1])
    U_train = U[:, :split_idx].T
    U_test = U[:, split_idx:].T
    return Nx, U_train, U_test
