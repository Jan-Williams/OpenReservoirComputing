"""Training functions for reservoir computer controllers."""

from typing import TypeVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from orc.control.base import RCControllerBase
from orc.control.models import ESNController
from orc.utils.regressions import (
    _solve_all_ridge_reg,
    _solve_all_ridge_reg_batched,
    ridge_regression,
)

VariableRCController = TypeVar(name="VariableRCController", bound=RCControllerBase)


def train_RCController(
    model: VariableRCController,
    train_seq: Array,
    control_seq: Array,
    target_seq: Array | None = None,
    spinup: int = 0,
    initial_res_state: Array | None = None,
    beta: float = 8e-8,
    batch_size: int | None = None,
    multi_sequence: bool = False,
    **force_kwargs,
) -> tuple[VariableRCController, Array]:
    """Unified training function for reservoir computer controllers.

    Works with any model inheriting from RCControllerBase, including
    ESNController and custom models with user-defined readout layers.

    Parameters
    ----------
    model : RCControllerBase
        Reservoir computer controller model to train.
    train_seq : Array
        Training input sequence for reservoir. Shape ``(seq_len, data_dim)``
        for a single trajectory, or ``(n_traj, seq_len, data_dim)`` when
        ``multi_sequence=True``. All trajectories must have the same length.
    control_seq : Array
        Control input sequence for reservoir. Shape ``(seq_len, control_dim)``
        for a single trajectory, or ``(n_traj, seq_len, control_dim)`` when
        ``multi_sequence=True``.
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
        Only used when readout.chunks > 0.
    multi_sequence : bool
        If True, treat train_seq, control_seq, and target_seq as batches of
        trajectories with a leading trajectory axis. Reservoir states from all
        trajectories are concatenated (after spinup) before solving the
        regression. This flag is a static boolean suitable for use as
        ``static_argnums`` when wrapping with ``jax.jit``.
    **force_kwargs
        Additional keyword arguments passed to model.force().

    Returns
    -------
    model : RCControllerBase
        Trained model.
    tot_res_seq : Array
        Full reservoir state sequence from teacher forcing. Shape
        ``(seq_len, ...)`` for single-sequence or ``(n_traj, seq_len, ...)``
        for multi-sequence.
    """
    if multi_sequence:
        if train_seq.shape[:2] != control_seq.shape[:2]:
            raise ValueError(
                "train_seq and control_seq must have the same (n_traj, seq_len)."
            )
    else:
        if train_seq.shape[0] != control_seq.shape[0]:
            raise ValueError("train_seq and control_seq must have the same length.")

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
        tot_control_seq = control_seq
        target_seq = train_seq[:, 1:, :] if multi_sequence else train_seq[1:, :]
    else:
        if multi_sequence:
            tot_seq = jnp.concatenate([train_seq, target_seq[:, -1:, :]], axis=1)
            tot_control_seq = jnp.concatenate(
                [control_seq, control_seq[:, -1:, :]], axis=1
            )
        else:
            tot_seq = jnp.vstack((train_seq, target_seq[-1:]))
            tot_control_seq = jnp.vstack((control_seq, control_seq[-1:]))

    if multi_sequence:
        tot_res_seq = eqx.filter_vmap(model.force)(
            tot_seq, tot_control_seq, initial_res_state
        )
        res_seq = tot_res_seq[:, :-1]
        res_seq_train = eqx.filter_vmap(model.readout.prepare_train)(res_seq)
        train_target = eqx.filter_vmap(model.readout.prepare_target)(target_seq)
        res_seq_train = res_seq_train[:, spinup:].reshape(
            (-1,) + res_seq_train.shape[2:]
        )
        train_target = train_target[:, spinup:].reshape((-1,) + train_target.shape[2:])
    else:
        tot_res_seq = model.force(
            tot_seq, tot_control_seq, initial_res_state, **force_kwargs
        )
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


def train_ESNController(
    model: ESNController,
    train_seq: Array,
    control_seq: Array,
    target_seq: Array | None = None,
    spinup: int = 0,
    initial_res_state: Array | None = None,
    beta: float = 8e-8,
    multi_sequence: bool = False,
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
    multi_sequence : bool
        If True, treat train_seq and control_seq as batches of trajectories
        with shape (n_traj, seq_len, ...). See train_RCController for details.

    Returns
    -------
    model : ESNController
        Trained ESN controller model.
    res_seq : Array
        Training sequence of reservoir states.
    """
    if not isinstance(model, ESNController):
        raise TypeError("Model must be an ESNController.")

    return train_RCController(
        model,
        train_seq,
        control_seq,
        target_seq,
        spinup,
        initial_res_state,
        beta,
        multi_sequence=multi_sequence,
    )
