"""Training functions for reservoir computer classifiers."""

from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from orc.classifier.base import RCClassifierBase
from orc.classifier.models import ESNClassifier
from orc.utils.regressions import (
    _solve_all_ridge_reg,
    _solve_all_ridge_reg_batched,
    ridge_regression,
)

VariableRCClassifier = TypeVar(name="VariableRCClassifier", bound=RCClassifierBase)


def train_RCClassifier(
    model: VariableRCClassifier,
    train_seqs: Array,
    labels: Array,
    spinup: int = 0,
    beta: float = 8e-8,
    batch_size: int | None = None,
    **force_kwargs,
) -> VariableRCClassifier:
    """Unified training function for reservoir computer classifiers.

    Works with any model inheriting from RCClassifierBase, including
    ESNClassifier and custom models with user-defined readout layers.

    Trains the classifier by forcing each input sequence through the reservoir,
    extracting a feature vector from the reservoir states, and solving ridge
    regression against one-hot encoded class labels.

    Parameters
    ----------
    model : RCClassifierBase
        Reservoir computer classifier model to train.
    train_seqs : Array
        Batch of training input sequences,
        (shape=(n_samples, seq_len, data_dim)).
    labels : Array
        Integer class labels for each sequence, (shape=(n_samples,)).
        Values should be in [0, n_classes).
    spinup : int
        Number of initial reservoir states to discard before extracting
        features. Only used when model.state_repr="mean".
    beta : float
        Tikhonov regularization parameter.
    batch_size : int, optional
        Number of parallel reservoirs to process in each batch for ridge
        regression. If None (default), processes all reservoirs at once.
        Only used when readout.chunks > 0.
    **force_kwargs
        Additional keyword arguments passed to model.force().

    Returns
    -------
    model : RCClassifierBase
        Trained classifier model.
    """
    if train_seqs.shape[0] != labels.shape[0]:
        raise ValueError("Number of training sequences must match number of labels.")

    n_samples = train_seqs.shape[0]
    initial_res_states = jnp.tile(model.driver.default_state(), (n_samples, 1))

    # Force all sequences through the reservoir in parallel via vmap
    all_res_seqs = jax.vmap(model.force)(train_seqs, initial_res_states)
    # all_res_seqs shape: (n_samples, seq_len, res_dim)

    # Extract feature vectors
    if model.state_repr == "final":
        feature_matrix = all_res_seqs[:, -1, :]  # (n_samples, res_dim)
    else:  # "mean"
        feature_matrix = jnp.mean(all_res_seqs[:, spinup:, :], axis=1)

    # Build one-hot target matrix
    one_hot_targets = jnp.zeros((n_samples, model.n_classes), dtype=model.dtype)
    one_hot_targets = one_hot_targets.at[jnp.arange(n_samples), labels].set(1.0)

    # Use readout's prepare_train/prepare_target for shape handling
    feature_train = model.readout.prepare_train(feature_matrix)
    train_target = model.readout.prepare_target(one_hot_targets)

    if model.readout.chunks > 0:
        if batch_size is None:
            cmat = _solve_all_ridge_reg(
                feature_train,
                train_target,
                beta,
            )
        else:
            cmat = _solve_all_ridge_reg_batched(
                feature_train,
                train_target,
                beta,
                batch_size,
            )
    else:
        cmat = ridge_regression(
            feature_train,
            train_target,
            beta,
        )

    new_readout = model.readout.set_wout(cmat)
    model = eqx.tree_at(lambda m: m.readout, model, new_readout)

    return model


def train_ESNClassifier(
    model: ESNClassifier,
    train_seqs: Array,
    labels: Array,
    spinup: int = 0,
    beta: float = 8e-8,
) -> ESNClassifier:
    """Training function for ESNClassifier.

    Trains the classifier by forcing each input sequence through the reservoir,
    extracting a feature vector from the reservoir states, and solving ridge
    regression against one-hot encoded class labels.

    Parameters
    ----------
    model : ESNClassifier
        ESNClassifier model to train.
    train_seqs : Array
        Batch of training input sequences,
        (shape=(n_samples, seq_len, data_dim)).
    labels : Array
        Integer class labels for each sequence, (shape=(n_samples,)).
        Values should be in [0, n_classes).
    spinup : int
        Number of initial reservoir states to discard before extracting
        features. Only used when model.state_repr="mean".
    beta : float
        Tikhonov regularization parameter.

    Returns
    -------
    model : ESNClassifier
        Trained ESN classifier model.
    """
    if not isinstance(model, ESNClassifier):
        raise TypeError("Model must be an ESNClassifier.")

    return train_RCClassifier(model, train_seqs, labels, spinup, beta)
