import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate

import orc


def test_esn_train():
    """
    Test forecast on Lorenz system. Passes if forecast is accurate for 100 steps.
    """
    esn = orc.models.ESN(data_dim=3, res_dim=2000, seed=0, bias=0)

    def lorenz(t, x, sigma=10, beta=8 / 3, rho=28):
        return np.array(
            [
                sigma * (x[1] - x[0]),
                x[0] * (rho - x[2]) - x[1],
                x[0] * x[1] - beta * x[2],
            ]
        )

    T = 600
    x0 = np.array([0.05, 1, 1.05])
    dt_data = 0.01
    xt_lorenz = integrate.solve_ivp(
        lorenz, [0, T], x0, method="RK45", t_eval=np.arange(0, T, dt_data), rtol=1e-12
    )
    U = xt_lorenz.y
    train_len = 50000
    jax_input = jax.numpy.array(U[:, :train_len]).T
    target_seq = jax.numpy.array(U[:, 1 : train_len + 1]).T.reshape(
        train_len, esn.readout.chunks, -1
    )
    esn, output_seq = orc.models.esn.train_ESN_forecaster(
        esn,
        jax_input,
        target_seq,
        spinup=500,
        initial_res_state=jax.numpy.zeros((1, 2000), dtype=jnp.float64),
        beta=8e-8,
    )
    fcast = esn.forecast(100, output_seq[-1])
    assert (
        jnp.linalg.norm(
            fcast - U[:, train_len + 1 : train_len + 101].T.reshape(100, -1)
        )
        / 100
        < 1e-3
    )


def test_periodic_par_esn(gen_KS_data):
    Nx = gen_KS_data[0]
    U_train = gen_KS_data[1]
    U_test = gen_KS_data[2]

    train_len = 15000
    input_sequence = jnp.array(U_train[-train_len - 1 : -1])
    target_sequence = jnp.array(U_train[-train_len:])
    NR = 1000
    chunks = 32
    locality = 3
    esn = orc.models.ESN(
        data_dim=Nx, res_dim=NR, seed=2, chunks=chunks, locality=locality
    )

    esn, output_seq = orc.models.esn.train_ESN_forecaster(
        esn,
        input_sequence,
        target_sequence,
        spinup=200,
        initial_res_state=jax.numpy.zeros((chunks, NR), dtype=jnp.float64),
        beta=8e-8,
    )

    fcast = esn.forecast(U_test.shape[0], output_seq[-1])

    assert jnp.linalg.norm(fcast[:50] - U_test[:50]) / (50 * Nx) < 1e-3

    esn = orc.models.ESN(
        data_dim=Nx,
        res_dim=NR,
        seed=2,
        chunks=chunks,
        locality=locality,
        periodic=False,
    )

    esn, output_seq = orc.models.esn.train_ESN_forecaster(
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
    U, t = KS_1D_PBC(u0, TN, dt, domain, Nx)  # Solve!

    # train-test split
    test_perc = 0.2
    split_idx = int((1 - test_perc) * U.shape[1])
    U_train = U[:, :split_idx].T
    U_test = U[:, split_idx:].T
    return Nx, U_train, U_test


def KS_1D_PBC(
    u0, tN, dt=0.1, domain=(0, 100), Nx=200, rtol=1e-12, atol=1e-12, max_steps=1e4
):
    # Define the spatial grid
    Nx = Nx - 1  # PBC - only solving for Nx-1 points
    u0 = u0[:-1]  # PBC: throw away duplicate point
    x = np.linspace(domain[0], domain[1], int(Nx), endpoint=False)
    dx = x[1] - x[0]

    # initialize solutions
    U = np.zeros((Nx, int(tN / dt)))
    U[:, 0] = u0

    # Define the wavenumbers
    k = np.fft.fftfreq(int(Nx), d=dx) * 2 * np.pi
    k2 = k**2
    k4 = k**4

    # define operators
    L_op = k2 - k4  # Linear operator
    def N_op_u(u):  # Nonlinear operator acts on u
        return 1j * k * np.fft.fft(-0.5 * u**2)
    def N_op_uhat(u):
        return 1j * k * np.fft.fft(-0.5 * np.real(np.fft.ifft(u)) ** 2)

    # compute exp coeffs using complex means from Kassam et al, 2005
    E1 = np.exp(L_op * dt)
    E2 = np.exp(L_op * dt / 2)
    M = 16
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * np.column_stack([L_op] * M) + np.row_stack([r] * (Nx))
    Q = dt * np.mean((np.exp(LR / 2) - 1) / LR, axis=1)
    f1 = dt * np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1)
    f2 = dt * np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1)
    f3 = dt * np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1)

    # time stepping loop
    for i in range(int(tN / dt) - 1):
        u = U[:, i]
        u_hat = np.fft.fft(u)
        a = E2 * u_hat + Q * N_op_u(u)
        b = E2 * u_hat + Q * N_op_uhat(a)
        c = E2 * a + Q * (2 * N_op_uhat(b) - N_op_u(u))
        u_hat = (
            E1 * u_hat
            + f1 * N_op_u(u)
            + f2 * (N_op_uhat(a) + N_op_uhat(b))
            + f3 * N_op_uhat(c)
        )

        U[:, i + 1] = np.real(np.fft.ifft(u_hat, n=Nx))

    # Fill in PBC
    U = np.vstack((U, U[0, :]))
    t = np.arange(0, tN, dt)

    return U, t
