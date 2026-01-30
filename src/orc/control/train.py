"""Training functions for reservoir computer controllers."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from orc.control.models import ESNController
from orc.readouts import NonlinearReadout
from orc.utils.regressions import ridge_regression


def train_ESNController(
    model: ESNController,
    train_seq: Array,
    control_seq: Array,
    target_seq: Array = None,
    spinup: int = 0,
    initial_res_state: Array = None,
    beta: float = 8e-8,
) -> tuple[ESNController, Array]:
    """Training function for ESNController.

    Parameters
    ----------
    model : ESNController
        ESNController model to train.
    train_seq : Array
        Training input sequence for reservoir, (shape=(seq_len, data_dim)).
    control_seq : Array
        Control input sequence for reservoir, (shape=(seq_len, control_dim)).
    target_seq : Array
        Target sequence for training reservoir, (shape=(seq_len, data_dim)).
        If None, defaults to train_seq[1:].
    initial_res_state : Array
        Initial reservoir state, (shape=(res_dim,)).
    spinup : int
        Initial transient of reservoir states to discard.
    beta : float
        Tikhonov regularization parameter.

    Returns
    -------
    model : ESNController
        Trained ESN controller model.
    res_seq : Array
        Training sequence of reservoir states.
    """
    # Check that model is an ESN Controller
    if not isinstance(model, ESNController):
        raise TypeError("Model must be an ESNController.")

    # check that train_seq and control_seq have the same length
    if train_seq.shape[0] != control_seq.shape[0]:
        raise ValueError("train_seq and control_seq must have the same length.")

    # check that spinup is less than the length of the training sequence
    if spinup >= train_seq.shape[0]:
        raise ValueError(
            "spinup must be less than the length of the training sequence."
        )

    if initial_res_state is None:
        initial_res_state = jnp.zeros(model.res_dim, dtype=model.dtype)

    if target_seq is None:
        tot_seq = train_seq
        tot_control_seq = control_seq
        target_seq = train_seq[1:, :]
        train_seq = train_seq[:-1, :]
        control_seq = control_seq[:-1, :]
    else:
        tot_seq = jnp.vstack((train_seq, target_seq[-1:]))
        tot_control_seq = control_seq

    tot_res_seq = model.force(tot_seq, tot_control_seq, initial_res_state)
    res_seq = tot_res_seq[:-1]
    if isinstance(model.readout, NonlinearReadout):
        res_seq_train = model.readout.nonlinear_transform(res_seq)
    else:
        res_seq_train = res_seq

    cmat = ridge_regression(res_seq_train[spinup:], target_seq[spinup:], beta)
    cmat = cmat.reshape(1, cmat.shape[0], cmat.shape[1])
    def where(m):
        return m.readout.wout

    model = eqx.tree_at(where, model, cmat)

    return model, tot_res_seq
