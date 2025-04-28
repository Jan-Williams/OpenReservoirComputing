import jax
import jax.numpy as jnp
import numpy as np
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
