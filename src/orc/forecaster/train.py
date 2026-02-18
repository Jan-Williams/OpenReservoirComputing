"""Training functions for reservoir computer forecasters."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from orc.forecaster.base import RCForecasterBase
from orc.forecaster.models import CESNForecaster, EnsembleESNForecaster, ESNForecaster
from orc.utils.regressions import (
    _solve_all_ridge_reg,
    _solve_all_ridge_reg_batched,
    ridge_regression,
)


def train_RCForecaster(
    model: RCForecasterBase,
    train_seq: Array,
    target_seq: Array = None,
    spinup: int = 0,
    initial_res_state: Array = None,
    beta: float = 8e-8,
    batch_size: int = None,
    **force_kwargs,
) -> tuple[RCForecasterBase, Array]:
    """Unified training function for reservoir computer forecasters.

    Works with any model inheriting from RCForecasterBase, including
    ESNForecaster, CESNForecaster, EnsembleESNForecaster, and custom
    models with user-defined readout layers.

    For continuous models (CESNForecaster), pass the time vector as
    ``ts=t_train`` via keyword arguments.

    Parameters
    ----------
    model : RCForecasterBase
        Reservoir computer forecaster model to train.
    train_seq : Array
        Training input sequence for reservoir, (shape=(seq_len, data_dim)).
    target_seq : Array, optional
        Target sequence for training reservoir, (shape=(seq_len, data_dim)).
        If None, defaults to train_seq[1:].
    spinup : int
        Initial transient of reservoir states to discard.
    initial_res_state : Array, optional
        Initial reservoir state. If None, uses model.driver.default_state().
    beta : float
        Tikhonov regularization parameter.
    batch_size : int, optional
        Number of parallel reservoirs to process in each batch for ridge
        regression. If None (default), processes all reservoirs at once.
        Use smaller values to reduce memory usage for large numbers of
        parallel reservoirs. Only used when readout.chunks > 0.
    **force_kwargs
        Additional keyword arguments passed to model.force() (e.g. ts=t_train
        for continuous models).

    Returns
    -------
    model : RCForecasterBase
        Trained model.
    tot_res_seq : Array
        Full sequence of reservoir states from teacher forcing.
    """
    if spinup >= train_seq.shape[0]:
        raise ValueError(
            "spinup must be less than the length of the training sequence."
        )

    if initial_res_state is None:
        initial_res_state = model.driver.default_state()

    if target_seq is None:
        tot_seq = train_seq
        target_seq = train_seq[1:, :]
        train_seq = train_seq[:-1, :]
    else:
        tot_seq = jnp.vstack((train_seq, target_seq[-1:]))

    tot_res_seq = model.force(tot_seq, initial_res_state, **force_kwargs)
    res_seq = tot_res_seq[:-1]

    res_seq_train = model.readout.prepare_train(res_seq)
    train_target = model.readout.prepare_target(target_seq)

    if model.readout.chunks > 0:
        if batch_size is None:
            cmat = _solve_all_ridge_reg(
                res_seq_train[spinup:],
                train_target[spinup:],
                beta,
            )
        else:
            cmat = _solve_all_ridge_reg_batched(
                res_seq_train[spinup:],
                train_target[spinup:],
                beta,
                batch_size,
            )
    else:
        cmat = ridge_regression(
            res_seq_train[spinup:],
            train_target[spinup:],
            beta,
        )

    new_readout = model.readout.set_wout(cmat)
    model = eqx.tree_at(lambda m: m.readout, model, new_readout)

    return model, tot_res_seq


def train_ESNForecaster(
    model: ESNForecaster,
    train_seq: Array,
    target_seq: Array = None,
    spinup: int = 0,
    initial_res_state: Array = None,
    beta: float = 8e-8,
    batch_size: int = None,
) -> tuple[ESNForecaster, Array]:
    """Training function for ESNForecaster.

    Parameters
    ----------
    model : ESNForecaster
        ESNForecaster model to train.
    train_seq : Array
        Training input sequence for reservoir, (shape=(seq_len, data_dim)).
    target_seq : Array
        Target sequence for training reservoir, (shape=(seq_len, data_dim)).
    initial_res_state : Array
        Initial reservoir state, (shape=(chunks, res_dim,)).
    spinup : int
        Initial transient of reservoir states to discard.
    beta : float
        Tikhonov regularization parameter.
    batch_size : int, optional
        Number of parallel reservoirs to process in each batch for ridge regression.
        If None (default), processes all reservoirs at once. Use smaller values
        to reduce memory usage for large numbers of parallel reservoirs.

    Returns
    -------
    model : ESNForecaster
        Trained ESN model.
    res_seq : Array
        Training sequence of reservoir states.
    """
    if not isinstance(model, ESNForecaster):
        raise TypeError("Model must be an ESNForecaster.")

    return train_RCForecaster(
        model, train_seq, target_seq, spinup, initial_res_state, beta, batch_size
    )


def train_CESNForecaster(
    model: CESNForecaster,
    train_seq: Array,
    t_train: Array,
    target_seq: Array = None,
    spinup: int = 0,
    initial_res_state: Array = None,
    beta: float = 8e-8,
    batch_size: int = None,
) -> tuple[CESNForecaster, Array]:
    """Training function for CESNForecaster.

    Parameters
    ----------
    model : CESNForecaster
        CESNForecaster model to train.
    train_seq : Array
        Training input sequence for reservoir, (shape=(seq_len, data_dim)).
    t_train : Array
        time vector corresponding to the training sequence, (shape=(seq_len,)).
    target_seq : Array
        Target sequence for training reservoir, (shape=(seq_len, data_dim)).
    initial_res_state : Array
        Initial reservoir state, (shape=(chunks, res_dim,)).
    spinup : int
        Initial transient of reservoir states to discard.
    beta : float
        Tikhonov regularization parameter.
    batch_size : int, optional
        Number of parallel reservoirs to process in each batch for ridge regression.
        If None (default), processes all reservoirs at once. Use smaller values
        to reduce memory usage for large numbers of parallel reservoirs.

    Returns
    -------
    model : CESNForecaster
        Trained CESN model.
    res_seq : Array
        Training sequence of reservoir states.
    """
    if not isinstance(model, CESNForecaster):
        raise TypeError("Model must be a CESNForecaster.")

    if train_seq.shape[0] != t_train.shape[0]:
        raise ValueError("train_seq and t_train must have the same length.")

    return train_RCForecaster(
        model, train_seq, target_seq, spinup, initial_res_state, beta, batch_size,
        ts=t_train,
    )


def train_EnsembleESNForecaster(
    model: EnsembleESNForecaster,
    train_seq: Array,
    target_seq: Array | None = None,
    spinup: int = 0,
    initial_res_state: Array | None = None,
    beta: float = 8e-8,
    batch_size: int | None = None,
) -> tuple[EnsembleESNForecaster, Array]:
    """Training function for EnsembleESNForecaster.

    Parameters
    ----------
    model : EnsembleESNForecaster
        EnsembleESNForecaster model to train.
    train_seq : Array
        Training input sequence for reservoir, (shape=(seq_len, data_dim)).
    target_seq : Array
        Target sequence for training reservoir, (shape=(seq_len, data_dim)).
    initial_res_state : Array
        Initial reservoir state, (shape=(chunks, res_dim,)).
    spinup : int
        Initial transient of reservoir states to discard.
    beta : float
        Tikhonov regularization parameter.
    batch_size : int, optional
        Number of parallel reservoirs to process in each batch for ridge regression.
        If None (default), processes all reservoirs at once. Use smaller values
        to reduce memory usage for large numbers of parallel reservoirs.

    Returns
    -------
    model : EnsembleESNForecaster
        Trained ensemble ESN model.
    res_seq : Array
        Training sequence of reservoir states.
    """
    if not isinstance(model, EnsembleESNForecaster):
        raise TypeError("Model must be an EnsembleESNForecaster.")

    return train_RCForecaster(
        model, train_seq, target_seq, spinup, initial_res_state, beta, batch_size
    )
