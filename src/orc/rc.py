"""Define base class for Reservoir Computers."""

from abc import ABC

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from orc.drivers import DriverBase
from orc.embeddings import EmbedBase
from orc.readouts import ReadoutBase


class RCForecasterBase(eqx.Module, ABC):
    """Base class for reservoir computer forecasters.

    Defines the interface for the reservoir computer which includes the driver,
    readout and embedding layers.

    Attributes
    ----------
    driver : DriverBase
        Driver layer of the reservoir computer.
    readout : ReadoutBase
        Readout layer of the reservoir computer.
    embedding : EmbedBase
        Embedding layer of the reservoir computer.
    in_dim : int
        Dimension of the input data.
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
    force(in_seq, res_state)
        Teacher forces the reservoir with the input sequence.
    set_readout(readout)
        Replaces the readout layer of the reservoir computer.
    set_embedding(embedding)
        Replaces the embedding layer of the reservoir computer.
    """

    driver: DriverBase
    readout: ReadoutBase
    embedding: EmbedBase
    in_dim: int
    out_dim: int
    res_dim: int
    dtype: Float = jnp.float64
    seed: int = 0

    @eqx.filter_jit
    def force(self, in_seq: Array, res_state: Array) -> Array:
        """Teacher forces the reservoir.

        Parameters
        ----------
        in_seq: Array
            Input sequence to force the reservoir, (shape=(seq_len, data_dim)).
        res_state : Array
            Initial reservoir stat, (shape=(res_dim,)).

        Returns
        -------
        Array
            Forced reservoir sequence, (shape=(seq_len, res_dim)).
        """

        def scan_fn(state, in_vars):
            proj_vars = self.embedding(in_vars)
            res_state = self.driver(proj_vars, state)
            return (res_state, res_state)

        _, res_seq = jax.lax.scan(scan_fn, res_state, in_seq)
        return res_seq

    def set_readout(self, readout: ReadoutBase):
        """Replace readout layer.

        Parameters
        ----------
        readout : ReadoutBase
            New readout layer.

        Returns
        -------
        RCForecasterBase
            Updated model with new readout layer.
        """

        def where(m: RCForecasterBase):
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
        RCForecasterBase
            Updated model with new embedding layer.
        """

        def where(m: RCForecasterBase):
            return m.embedding

        new_model = eqx.tree_at(where, self, embedding)
        return new_model

    @eqx.filter_jit
    def forecast(self, fcast_len: int, res_state: Array) -> Array:
        """Forecast from an initial reservoir state.

        Parameters
        ----------
        fcast_len : int
            Steps to forecast.
        res_state : Array
            Initial reservoir state, (shape=(res_dim)).

        Returns
        -------
        Array
            Forecasted states, (shape=(fcast_len, data_dim))
        """

        def scan_fn(state, _):
            out_state = self.driver(self.embedding(self.readout(state)), state)
            return (out_state, self.readout(out_state))

        _, state_seq = jax.lax.scan(scan_fn, res_state, None, length=fcast_len)
        return state_seq