"""Training functions for reservoir computer forecasters."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from orc.forecaster.models import CESNForecaster, EnsembleESNForecaster, ESNForecaster
from orc.readouts import ParallelNonlinearReadout
from orc.utils.regressions import (
    _solve_all_ridge_reg,
    _solve_all_ridge_reg_batched,
)


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
    # Check that model is an ESN
    if not isinstance(model, ESNForecaster):
        raise TypeError("Model must be an ESNForecaster.")

    # check that spinup is less than the length of the training sequence
    if spinup >= train_seq.shape[0]:
        raise ValueError(
            "spinup must be less than the length of the training sequence."
        )

    if initial_res_state is None:
        initial_res_state = jnp.zeros(
            (
                model.embedding.chunks,
                model.res_dim,
            ),
            dtype=model.dtype,
        )

    if target_seq is None:
        tot_seq = train_seq
        target_seq = train_seq[1:, :]
        train_seq = train_seq[:-1, :]
    else:
        tot_seq = jnp.vstack((train_seq, target_seq[-1:]))

    tot_res_seq = model.force(tot_seq, initial_res_state)
    res_seq = tot_res_seq[:-1]
    if isinstance(model.readout, ParallelNonlinearReadout):
        res_seq_train = eqx.filter_vmap(model.readout.nonlinear_transform)(res_seq)
    else:
        res_seq_train = res_seq

    if batch_size is None:
        cmat = _solve_all_ridge_reg(
            res_seq_train[spinup:],
            target_seq[spinup:].reshape(
                res_seq[spinup:].shape[0], res_seq.shape[1], -1
            ),
            beta,
        )
    else:
        cmat = _solve_all_ridge_reg_batched(
            res_seq_train[spinup:],
            target_seq[spinup:].reshape(
                res_seq[spinup:].shape[0], res_seq.shape[1], -1
            ),
            beta,
            batch_size,
        )

    def where(m):
        return m.readout.wout

    model = eqx.tree_at(where, model, cmat)

    return model, tot_res_seq


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
    # check that model is continuous
    if not isinstance(model, CESNForecaster):
        raise TypeError("Model must be a CESNForecaster.")

    # check that train_seq and t_train have the same length
    if train_seq.shape[0] != t_train.shape[0]:
        raise ValueError("train_seq and t_train must have the same length.")

    # check that spinup is less than the length of the training sequence
    if spinup >= train_seq.shape[0]:
        raise ValueError(
            "spinup must be less than the length of the training sequence."
        )

    if initial_res_state is None:
        initial_res_state = jnp.zeros(
            (
                model.embedding.chunks,
                model.res_dim,
            ),
            dtype=model.dtype,
        )

    if target_seq is None:
        tot_seq = train_seq
        target_seq = train_seq[1:, :]
        train_seq = train_seq[:-1, :]
    else:
        tot_seq = jnp.vstack((train_seq, target_seq[-1:]))

    tot_res_seq = model.force(tot_seq, initial_res_state, ts=t_train)
    res_seq = tot_res_seq[:-1]
    if isinstance(model.readout, ParallelNonlinearReadout):
        res_seq_train = eqx.filter_vmap(model.readout.nonlinear_transform)(res_seq)
    else:
        res_seq_train = res_seq

    if batch_size is None:
        cmat = _solve_all_ridge_reg(
            res_seq_train[spinup:],
            target_seq[spinup:].reshape(
                res_seq[spinup:].shape[0], res_seq.shape[1], -1
            ),
            beta,
        )
    else:
        cmat = _solve_all_ridge_reg_batched(
            res_seq_train[spinup:],
            target_seq[spinup:].reshape(
                res_seq[spinup:].shape[0], res_seq.shape[1], -1
            ),
            beta,
            batch_size,
        )

    def where(m):
        return m.readout.wout

    model = eqx.tree_at(where, model, cmat)

    return model, tot_res_seq


def train_EnsembleESNForecaster(
    model: EnsembleESNForecaster,
    train_seq: Array,
    target_seq: Array | None = None,
    spinup: int = 0,
    initial_res_state: Array | None = None,
    beta: float = 8e-8,
    batch_size: int | None = None,
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
    # Check that model is an ESN
    if not isinstance(model, EnsembleESNForecaster):
        raise TypeError("Model must be an EnsembleESNForecaster.")

    # check that spinup is less than the length of the training sequence
    if spinup >= train_seq.shape[0]:
        raise ValueError(
            "spinup must be less than the length of the training sequence."
        )

    if initial_res_state is None:
        initial_res_state = jnp.zeros(
            (
                model.embedding.chunks,
                model.res_dim,
            ),
            dtype=model.dtype,
        )

    if target_seq is None:
        tot_seq = train_seq
        target_seq = train_seq[1:, :]
        train_seq = train_seq[:-1, :]
    else:
        tot_seq = jnp.vstack((train_seq, target_seq[-1:]))

    tot_res_seq = model.force(tot_seq, initial_res_state)
    res_seq = tot_res_seq[:-1]
    res_seq_train = res_seq

    repeated_target_seq = jnp.repeat(target_seq[:, None, :], model.chunks, axis=1)
    if batch_size is None:
        cmat = _solve_all_ridge_reg(
            res_seq_train[spinup:],
            repeated_target_seq[spinup:],
            beta,
        )
    else:
        cmat = _solve_all_ridge_reg_batched(
            res_seq_train[spinup:],
            repeated_target_seq[spinup:],
            beta,
            batch_size,
        )

    def where(m):
        return m.readout.wout

    model = eqx.tree_at(where, model, cmat)

    return model, tot_res_seq
