"""Define base class for Reservoir Computers."""

from abc import ABC, abstractmethod
from typing import Tuple

import jax
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from orc.drivers import DriverBase
from orc.readouts import ReadoutBase
from orc.embeddings import EmbedBase

class ReservoirComputerBase(eqx.Module, ABC):
    """
    Base class for Reservoir Computers. Defines the interface for the reservoir computer which 
    includes the driver, readout and embedding layers.

    Attributes
    ----------
    driver : DriverBase
        Driver layer of the reservoir computer.
    readout : ReadoutBase
        Readout layer of the reservoir computer.
    embedding : EmbedBase
        Embedding layer of the reservoir computer.
    data_dim : int
        Dimension of the input data.
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
    forecast(fcast_len, res_state)
        Forecasts the next fcast_len steps from a given intial reservoir state.
    set_readout(readout)
        Replaces the readout layer of the reservoir computer.
    set_embedding(embedding)
        Replaces the embedding layer of the reservoir computer.
    """

    driver: DriverBase
    readout: ReadoutBase
    embedding: EmbedBase
    data_dim: int
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
            proj_vars = self.embedding.embed(in_vars)
            res_state = self.driver.advance(proj_vars, state)
            return (res_state, res_state)

        _, res_seq = jax.lax.scan(scan_fn, res_state, in_seq)
        return res_seq
    
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
            out_state = self.driver.advance(
                self.embedding.embed(self.readout.readout(state)), state
            )
            return (out_state, self.readout.readout(out_state))

        _, state_seq = jax.lax.scan(scan_fn, res_state, None, length=fcast_len)
        return state_seq
    
    def set_readout(self, readout: ReadoutBase):
        """Replace readout layer.

        Parameters
        ----------
        readout : ReadoutBase
            New readout layer.

        Returns
        -------
        ESN
            Updated model with new readout layer.
        """

        def where(m: ReservoirComputerBase):
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
        ReservoirComputerBase
            Updated model with new embedding layer.
        """

        def where(m: ReservoirComputerBase):
            return m.embedding

        new_model = eqx.tree_at(where, self, embedding)
        return new_model
    
    
def train_RC_forecaster(
    model: ReservoirComputerBase,
    train_seq: Array,
    spinup: int = 0,
    initial_res_state: Array = None,
    beta: float = 8e-8,
) -> Tuple[ReservoirComputerBase, Array]:
    """Training function for RC forecaster.

    Parameters
    ----------
    model : ReservoirComputerBase
        ReservoirComputerBase model to train.
    in_seq : Array
        Training sequence for reservoir, (shape=(seq_len, data_dim)).
    initial_res_state : Array
        Initial reservoir state, (shape=(res_dim,)).
    spinup : int
        Initial transient of reservoir states to discard. 
    beta : float
        Tikhonov regularization parameter.

    Returns
    -------
    model : ReservoirComputerBase
        Trained ReservoirComputerBase model.
    """

    # zero IC of RC if not provided
    if initial_res_state is None:
        initial_res_state = jnp.zeros((model.res_dim,), dtype=model.dtype)

    # force the reservoir
    res_seq = model.force(train_seq[:-1,:], initial_res_state)

    # solve ridge regression problem to train readout
    lhs = res_seq[spinup:].T @ res_seq[spinup:] + beta * jnp.eye(
        model.res_dim, dtype=model.dtype
    )
    rhs = res_seq[spinup:].T @ train_seq[spinup+1:,:]  
    cmat = jax.scipy.linalg.solve(lhs, rhs, assume_a="sym").T

    # replace wout with learned weights
    def where(m):
        return m.readout.wout

    model = eqx.tree_at(where, model, cmat)
    return model, res_seq