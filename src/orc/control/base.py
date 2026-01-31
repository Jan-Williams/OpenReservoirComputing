"""Defines base classes for Reservoir Computer Controllers."""

from abc import ABC

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from orc.drivers import DriverBase
from orc.embeddings import EmbedBase
from orc.readouts import ReadoutBase


class RCControllerBase(eqx.Module, ABC):
    """Base class for reservoir computer controllers.

    Defines the interface for the reservoir computer controller which includes
    the driver, readout and embedding layers. Unlike the forecaster, the controller
    handles an additional control input at each time step.

    Attributes
    ----------
    driver : DriverBase
        Driver layer of the reservoir computer.
    readout : ReadoutBase
        Readout layer of the reservoir computer.
    embedding : EmbedBase
        Embedding layer of the reservoir computer. Should accept concatenated
        [input, control] vectors.
    in_dim : int
        Dimension of the system input data.
    control_dim : int
        Dimension of the control input.
    out_dim : int
        Dimension of the output data.
    res_dim : int
        Dimension of the reservoir.
    dtype : type
        Data type of the reservoir computer (jnp.float64 is highly recommended).
    seed : int
        Random seed for generating the PRNG key for the reservoir computer.


    Methods
    -------
    force(in_seq, control_seq, res_state)
        Teacher forces the reservoir with input and control sequences.
    apply_control(control_seq, fcast_len, res_state)
        Apply a predefined control sequence in closed-loop.
    set_readout(readout)
        Replaces the readout layer of the reservoir computer.
    set_embedding(embedding)
        Replaces the embedding layer of the reservoir computer.
    """

    driver: DriverBase
    readout: ReadoutBase
    embedding: EmbedBase
    in_dim: int
    control_dim: int
    out_dim: int
    res_dim: int
    dtype: Float = jnp.float64
    seed: int = 0

    def __init__(
        self,
        driver: DriverBase,
        readout: ReadoutBase,
        embedding: EmbedBase,
        in_dim: int,
        control_dim: int,
        dtype: Float = jnp.float64,
        seed: int = 0,
    ) -> None:
        """Initialize RCController Base.

        Parameters
        ----------
        driver : DriverBase
            Driver layer of the reservoir computer.
        readout : ReadoutBase
            Readout layer of the reservoir computer.
        embedding : EmbedBase
            Embedding layer of the reservoir computer.
        in_dim : int
            Dimension of the system input data.
        control_dim : int
            Dimension of the control input.
        dtype : type
            Data type of the reservoir computer (jnp.float64 is highly recommended).
        seed : int
            Random seed for generating the PRNG key for the reservoir computer.
        """
        self.driver = driver
        self.readout = readout
        self.embedding = embedding
        self.in_dim = in_dim
        self.control_dim = control_dim
        self.out_dim = self.readout.out_dim
        self.res_dim = self.driver.res_dim
        self.dtype = dtype
        self.seed = seed

    @eqx.filter_jit
    def force(self, in_seq: Array, control_seq: Array, res_state: Array) -> Array:
        """Teacher forces the reservoir with input and control sequences.

        Parameters
        ----------
        in_seq: Array
            Input sequence to force the reservoir, (shape=(seq_len, in_dim)).
        control_seq: Array
            Control sequence to force the reservoir, (shape=(seq_len, control_dim)).
        res_state : Array
            Initial reservoir state, (shape=(res_dim,)).

        Returns
        -------
        Array
            Forced reservoir sequence, (shape=(seq_len, res_dim)).
        """

        def scan_fn(state, in_vars):
            in_state, control_state = in_vars
            # Concatenate input and control for embedding
            combined_input = jnp.concatenate([in_state, control_state])
            proj_vars = self.embedding.embed(combined_input)
            res_state = self.driver.advance(proj_vars, state)
            return (res_state, res_state)

        _, res_seq = jax.lax.scan(scan_fn, res_state, (in_seq, control_seq))
        return res_seq

    def __call__(
        self, in_seq: Array, control_seq: Array, res_state: Array
    ) -> Array:
        """Teacher forces the reservoir, wrapper for `force` method.

        Parameters
        ----------
        in_seq: Array
            Input sequence to force the reservoir, (shape=(seq_len, in_dim)).
        control_seq: Array
            Control sequence to force the reservoir, (shape=(seq_len, control_dim)).
        res_state : Array
            Initial reservoir state, (shape=(res_dim,)).

        Returns
        -------
        Array
            Forced reservoir sequence, (shape=(seq_len, res_dim)).
        """
        return self.force(in_seq, control_seq, res_state)

    @eqx.filter_jit
    def apply_control(
        self, control_seq: Array, fcast_len: int, res_state: Array
    ) -> Array:
        """Apply a predefined control sequence in closed-loop.

        The readout feeds back as the next input: u(t+1) = readout(x(t)).
        Control c(t) comes from the provided control_seq.

        Parameters
        ----------
        control_seq : Array
            Control sequence to apply, (shape=(fcast_len, control_dim)).
        fcast_len : int
            Number of steps to apply control.
        res_state : Array
            Initial reservoir state, (shape=(res_dim,)).

        Returns
        -------
        Array
            Controlled output trajectory, (shape=(fcast_len, out_dim)).
        """

        def scan_fn(state, control_vars):
            # Get output from current reservoir state
            out_state = self.readout(state)
            # Concatenate output (as next input) with control
            combined_input = jnp.concatenate([out_state, control_vars])
            # Embed and advance reservoir
            proj_vars = self.embedding(combined_input)
            next_res_state = self.driver(proj_vars, state)
            return (next_res_state, self.readout(next_res_state))

        _, state_seq = jax.lax.scan(scan_fn, res_state, control_seq[0:-1])
        # Prepend the initial state output
        pre_append_state = self.readout(res_state)
        return jnp.vstack([pre_append_state, state_seq])

    def set_readout(self, readout: ReadoutBase):
        """Replace readout layer.

        Parameters
        ----------
        readout : ReadoutBase
            New readout layer.

        Returns
        -------
        RCControllerBase
            Updated model with new readout layer.
        """

        def where(m: "RCControllerBase"):
            return m.readout

        new_model = eqx.tree_at(where, self, readout)
        return new_model

    def set_embedding(self, embedding: EmbedBase):
        """Replace embedding layer.

        Parameters
        ----------
        embedding : EmbedBase
            New embedding layer.

        Returns
        -------
        RCControllerBase
            Updated model with new embedding layer.
        """

        def where(m: "RCControllerBase"):
            return m.embedding

        new_model = eqx.tree_at(where, self, embedding)
        return new_model
