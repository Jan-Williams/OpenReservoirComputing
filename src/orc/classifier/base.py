"""Defines base classes for Reservoir Computer Classifiers."""

from abc import ABC

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from orc.drivers import DriverBase
from orc.embeddings import EmbedBase
from orc.readouts import ReadoutBase


class RCClassifierBase(eqx.Module, ABC):
    """Base class for reservoir computer classifiers.

    Defines the interface for the reservoir computer classifier which includes
    the driver, readout and embedding layers. The classifier forces input sequences
    through the reservoir, extracts a feature vector from the reservoir states,
    and applies a trained readout to produce class probabilities.

    Attributes
    ----------
    driver : DriverBase
        Driver layer of the reservoir computer.
    readout : ReadoutBase
        Readout layer of the reservoir computer. Output dimension equals n_classes.
    embedding : EmbedBase
        Embedding layer of the reservoir computer.
    in_dim : int
        Dimension of the input data.
    out_dim : int
        Dimension of the output data (equals n_classes).
    res_dim : int
        Dimension of the reservoir.
    n_classes : int
        Number of classification classes.
    state_repr : str
        Reservoir state representation for classification.
        "final" uses the last reservoir state, "mean" averages states after spinup.
    dtype : type
        Data type of the reservoir computer (jnp.float64 is highly recommended).
    seed : int
        Random seed for generating the PRNG key for the reservoir computer.

    Methods
    -------
    force(in_seq, res_state)
        Teacher forces the reservoir with the input sequence.
    classify(in_seq, res_state)
        Classify an input sequence, returning class probabilities.
    set_readout(readout)
        Replaces the readout layer of the reservoir computer.
    set_embedding(embedding)
        Replaces the embedding layer of the reservoir computer.
    """

    driver: DriverBase
    readout: ReadoutBase
    embedding: EmbedBase
    in_dim: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    res_dim: int = eqx.field(static=True)
    n_classes: int = eqx.field(static=True)
    state_repr: str = eqx.field(default="final", static=True)
    dtype: type = eqx.field(default=jnp.float64, static=True)
    seed: int = eqx.field(default=0, static=True)

    def __init__(
        self,
        driver: DriverBase,
        readout: ReadoutBase,
        embedding: EmbedBase,
        n_classes: int,
        state_repr: str = "final",
        dtype: type = jnp.float64,
        seed: int = 0,
    ) -> None:
        """Initialize RCClassifier Base.

        Parameters
        ----------
        driver : DriverBase
            Driver layer of the reservoir computer.
        readout : ReadoutBase
            Readout layer of the reservoir computer.
        embedding : EmbedBase
            Embedding layer of the reservoir computer.
        n_classes : int
            Number of classification classes.
        state_repr : str
            Reservoir state representation for classification.
            "final" uses the last reservoir state, "mean" averages states after spinup.
        dtype : type
            Data type of the reservoir computer (jnp.float64 is highly recommended).
        seed : int
            Random seed for generating the PRNG key for the reservoir computer.
        """
        if state_repr not in ("final", "mean"):
            raise ValueError(
                f"state_repr must be 'final' or 'mean', got '{state_repr}'."
            )
        self.driver = driver
        self.readout = readout
        self.embedding = embedding
        self.in_dim = self.embedding.in_dim
        self.out_dim = self.readout.out_dim
        self.res_dim = self.driver.res_dim
        self.n_classes = n_classes
        self.state_repr = state_repr
        self.dtype = dtype
        self.seed = seed

    @eqx.filter_jit
    def force(self, in_seq: Array, res_state: Array) -> Array:
        """Teacher forces the reservoir with the input sequence.

        Parameters
        ----------
        in_seq : Array
            Input sequence to force the reservoir, (shape=(seq_len, in_dim)).
        res_state : Array
            Initial reservoir state, (shape=(res_dim,)).

        Returns
        -------
        Array
            Forced reservoir sequence, (shape=(seq_len, res_dim)).
        """

        def scan_fn(state, in_vars):
            proj_vars = self.embedding.embed(in_vars)
            res_state = self.driver.advance(proj_vars, state)
            return (res_state, res_state)

        _, res_seq = jax.lax.scan(scan_fn, res_state, in_seq)
        return res_seq

    @eqx.filter_jit
    def classify(
        self, in_seq: Array, res_state: Array | None = None, spinup: int = 0
    ) -> Array:
        """Classify an input sequence.

        Forces the reservoir with the input sequence, extracts a feature vector
        from the reservoir states, and returns softmax class probabilities.

        Parameters
        ----------
        in_seq : Array
            Input sequence to classify, (shape=(seq_len, in_dim)).
        res_state : Array
            Initial reservoir state, (shape=(res_dim,)).
        spinup : int
            Number of initial reservoir states to discard before extracting
            features. Only used when state_repr="mean".

        Returns
        -------
        Array
            Class probabilities, (shape=(n_classes,)).
        """
        if res_state is None:
            res_state = jnp.zeros(self.res_dim)

        res_seq = self.force(in_seq, res_state)

        if self.state_repr == "final":
            feature = res_seq[-1]
        else:  # "mean"
            feature = jnp.mean(res_seq[spinup:], axis=0)

        logits = self.readout.readout(feature)
        return jax.nn.softmax(logits)

    def __call__(self, in_seq: Array, res_state: Array, spinup: int = 0) -> Array:
        """Classify an input sequence, wrapper for `classify` method.

        Parameters
        ----------
        in_seq : Array
            Input sequence to classify, (shape=(seq_len, in_dim)).
        res_state : Array
            Initial reservoir state, (shape=(res_dim,)).
        spinup : int
            Number of initial reservoir states to discard before extracting
            features. Only used when state_repr="mean".

        Returns
        -------
        Array
            Class probabilities, (shape=(n_classes,)).
        """
        return self.classify(in_seq, res_state, spinup)

    def set_readout(self, readout: ReadoutBase) -> "RCClassifierBase":
        """Replace readout layer.

        Parameters
        ----------
        readout : ReadoutBase
            New readout layer.

        Returns
        -------
        RCClassifierBase
            Updated model with new readout layer.
        """

        def where(m: "RCClassifierBase"):
            return m.readout

        new_model = eqx.tree_at(where, self, readout)
        return new_model

    def set_embedding(self, embedding: EmbedBase) -> "RCClassifierBase":
        """Replace embedding layer.

        Parameters
        ----------
        embedding : EmbedBase
            New embedding layer.

        Returns
        -------
        RCClassifierBase
            Updated model with new embedding layer.
        """

        def where(m: "RCClassifierBase"):
            return m.embedding

        new_model = eqx.tree_at(where, self, embedding)
        return new_model
