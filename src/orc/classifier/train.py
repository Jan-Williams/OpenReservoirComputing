"""Training functions for reservoir computer classifiers."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from orc.classifier.models import ESNClassifier
from orc.readouts import NonlinearReadout
from orc.utils.regressions import ridge_regression


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

    if train_seqs.shape[0] != labels.shape[0]:
        raise ValueError("Number of training sequences must match number of labels.")

    n_samples = train_seqs.shape[0]
    initial_res_states = jnp.zeros((n_samples, model.res_dim), dtype=model.dtype)

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

    # Apply optional nonlinear transform
    if isinstance(model.readout, NonlinearReadout):
        feature_matrix = model.readout.nonlinear_transform(feature_matrix)

    # Solve ridge regression
    cmat = ridge_regression(feature_matrix, one_hot_targets, beta)
    cmat = cmat.reshape(1, cmat.shape[0], cmat.shape[1])

    def where(m):
        return m.readout.wout

    model = eqx.tree_at(where, model, cmat)

    return model
