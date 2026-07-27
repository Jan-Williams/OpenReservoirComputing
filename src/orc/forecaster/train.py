"""Training functions for reservoir computer forecasters."""

from typing import TypeVar

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

VariableRCForecaster = TypeVar(name="VariableRCForecaster", bound=RCForecasterBase)


def train_RCForecaster(
    model: VariableRCForecaster,
    train_seq: Array,
    target_seq: Array | None = None,
    spinup: int = 0,
    initial_res_state: Array | None = None,
    beta: float = 8e-8,
    batch_size: int | None = None,
    multi_sequence: bool = False,
    **force_kwargs,
) -> tuple[VariableRCForecaster, Array]:
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
        Training input sequence for reservoir. Shape ``(seq_len, data_dim)``
        for a single trajectory, or ``(n_traj, seq_len, data_dim)`` when
        ``multi_sequence=True``. All trajectories must have the same length.
    target_seq : Array, optional
        Target sequence for training reservoir. Shape ``(seq_len, data_dim)``
        for a single trajectory, or ``(n_traj, seq_len, data_dim)`` when
        ``multi_sequence=True``. If None, defaults to train_seq[1:] (or
        train_seq[:, 1:] for multi-sequence).
    spinup : int
        Initial transient of reservoir states to discard per trajectory.
    initial_res_state : Array, optional
        Initial reservoir state. If None, uses model.driver.default_state()
        (tiled across trajectories when ``multi_sequence=True``). When
        ``multi_sequence=True`` and not None, must have a leading trajectory
        axis, i.e. shape ``(n_traj, ...)``.
    beta : float
        Tikhonov regularization parameter.
    batch_size : int, optional
        Number of parallel reservoirs to process in each batch for ridge
        regression. If None (default), processes all reservoirs at once.
        Use smaller values to reduce memory usage for large numbers of
        parallel reservoirs. Only used when readout.chunks > 0.
    multi_sequence : bool
        If True, treat train_seq (and target_seq) as a batch of trajectories
        with a leading trajectory axis. Reservoir states from all trajectories
        are concatenated (after spinup) before solving the regression. This
        flag is a static boolean suitable for use as ``static_argnums`` when
        wrapping with ``jax.jit``.
    **force_kwargs
        Additional keyword arguments passed to model.force() (e.g. ts=t_train
        for continuous models). When ``multi_sequence=True``, array-valued
        kwargs must have a leading trajectory axis (e.g. ts of shape
        ``(n_traj, seq_len)``).

    Returns
    -------
    model : RCForecasterBase
        Trained model.
    tot_res_seq : Array
        Full reservoir state sequence from teacher forcing. Shape
        ``(seq_len, ...)`` for single-sequence or ``(n_traj, seq_len, ...)``
        for multi-sequence.
    """
    seq_len = train_seq.shape[1] if multi_sequence else train_seq.shape[0]
    if spinup >= seq_len:
        raise ValueError(
            "spinup must be less than the length of the training sequence."
        )

    if initial_res_state is None:
        default = model.driver.default_state()
        if multi_sequence:
            n_traj = train_seq.shape[0]
            initial_res_state = jnp.broadcast_to(
                default[None], (n_traj,) + default.shape
            )
        else:
            initial_res_state = default

    if target_seq is None:
        tot_seq = train_seq
        target_seq = train_seq[:, 1:, :] if multi_sequence else train_seq[1:, :]
    else:
        if multi_sequence:
            tot_seq = jnp.concatenate([train_seq, target_seq[:, -1:, :]], axis=1)
        else:
            tot_seq = jnp.vstack((train_seq, target_seq[-1:]))

    if multi_sequence:
        if force_kwargs:
            ts = force_kwargs["ts"]

            def _force(seq, state, t):
                return model.force(seq, state, ts=t)  # ty: ignore[unknown-argument]

            tot_res_seq = eqx.filter_vmap(_force)(tot_seq, initial_res_state, ts)
        else:
            tot_res_seq = eqx.filter_vmap(model.force)(tot_seq, initial_res_state)
        res_seq = tot_res_seq[:, :-1]
        res_seq_train = eqx.filter_vmap(model.readout.prepare_train)(res_seq)
        train_target = eqx.filter_vmap(model.readout.prepare_target)(target_seq)
        res_seq_train = res_seq_train[:, spinup:].reshape(
            (-1,) + res_seq_train.shape[2:]
        )
        train_target = train_target[:, spinup:].reshape((-1,) + train_target.shape[2:])
    else:
        tot_res_seq = model.force(tot_seq, initial_res_state, **force_kwargs)
        res_seq = tot_res_seq[:-1]
        res_seq_train = model.readout.prepare_train(res_seq)
        train_target = model.readout.prepare_target(target_seq)
        res_seq_train = res_seq_train[spinup:]
        train_target = train_target[spinup:]

    if model.readout.chunks > 0:
        if batch_size is None:
            cmat = _solve_all_ridge_reg(res_seq_train, train_target, beta)
        else:
            cmat = _solve_all_ridge_reg_batched(
                res_seq_train, train_target, beta, batch_size
            )
    else:
        cmat = ridge_regression(res_seq_train, train_target, beta)

    new_readout = model.readout.set_wout(cmat)
    model = eqx.tree_at(lambda m: m.readout, model, new_readout)

    return model, tot_res_seq


def train_ESNForecaster(
    model: ESNForecaster,
    train_seq: Array,
    target_seq: Array | None = None,
    spinup: int = 0,
    initial_res_state: Array | None = None,
    beta: float = 8e-8,
    batch_size: int | None = None,
    multi_sequence: bool = False,
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
    multi_sequence : bool
        If True, treat train_seq as a batch of trajectories with shape
        (n_traj, seq_len, data_dim). See train_RCForecaster for details.

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
        model,
        train_seq,
        target_seq,
        spinup,
        initial_res_state,
        beta,
        batch_size,
        multi_sequence,
    )


def train_CESNForecaster(
    model: CESNForecaster,
    train_seq: Array,
    t_train: Array,
    target_seq: Array | None = None,
    spinup: int = 0,
    initial_res_state: Array | None = None,
    beta: float = 8e-8,
    batch_size: int | None = None,
    multi_sequence: bool = False,
) -> tuple[CESNForecaster, Array]:
    """Training function for CESNForecaster.

    Parameters
    ----------
    model : CESNForecaster
        CESNForecaster model to train.
    train_seq : Array
        Training input sequence for reservoir, (shape=(seq_len, data_dim)) or
        (shape=(n_traj, seq_len, data_dim)) when multi_sequence=True.
    t_train : Array
        Time vector corresponding to the training sequence, (shape=(seq_len,))
        or (shape=(n_traj, seq_len)) when multi_sequence=True.
    target_seq : Array
        Target sequence for training reservoir, (shape=(seq_len, data_dim)) or
        (shape=(n_traj, seq_len, data_dim)) when multi_sequence=True.
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
    multi_sequence : bool
        If True, treat train_seq as a batch of trajectories. t_train must then
        have shape (n_traj, seq_len). See train_RCForecaster for details.

    Returns
    -------
    model : CESNForecaster
        Trained CESN model.
    res_seq : Array
        Training sequence of reservoir states.
    """
    if not isinstance(model, CESNForecaster):
        raise TypeError("Model must be a CESNForecaster.")

    if multi_sequence:
        if train_seq.shape[:2] != t_train.shape[:2]:
            raise ValueError(
                "train_seq and t_train must have the same (n_traj, seq_len)."
            )
    else:
        if train_seq.shape[0] != t_train.shape[0]:
            raise ValueError("train_seq and t_train must have the same length.")

    return train_RCForecaster(
        model,
        train_seq,
        target_seq,
        spinup,
        initial_res_state,
        beta,
        batch_size,
        multi_sequence,
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
    multi_sequence: bool = False,
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
    multi_sequence : bool
        If True, treat train_seq as a batch of trajectories with shape
        (n_traj, seq_len, data_dim). See train_RCForecaster for details.

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
        model,
        train_seq,
        target_seq,
        spinup,
        initial_res_state,
        beta,
        batch_size,
        multi_sequence,
    )
