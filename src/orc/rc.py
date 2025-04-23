"""Define base class for Reservoir Computers."""

from abc import ABC

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from orc.drivers import DriverBase
from orc.embeddings import EmbedBase
from orc.readouts import ReadoutBase


class ReservoirComputerBase(eqx.Module, ABC):
    """Base class for Reservoir Computers.

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
            proj_vars = self.embedding(in_vars)
            res_state = self.driver(proj_vars, state)
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
            out_state = self.driver(
                self.embedding(self.readout(state)), state
            )
            return (out_state, self.readout(out_state))

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
    target_seq: Array,
    spinup: int = 0,
    initial_res_state: Array = None,
    beta: float = 8e-8,
) -> tuple[ReservoirComputerBase, Array]:
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
        initial_res_state = jnp.zeros((1, model.res_dim,), dtype=model.dtype)

    # force the reservoir
    res_seq = model.force(train_seq, initial_res_state)


    def solve_single_ridge_reg(res_seq, target_seq, beta):
        lhs = res_seq.T @ res_seq + beta * jnp.eye(
            res_seq.shape[1], dtype=res_seq.dtype
        )
        rhs = res_seq.T @ target_seq
        cmat = jax.scipy.linalg.solve(lhs, rhs, assume_a="sym").T
        return cmat
    

    solve_all_ridge_reg = eqx.filter_vmap(solve_single_ridge_reg, in_axes=eqx.if_array(1))
    cmat = solve_all_ridge_reg(res_seq[spinup:], target_seq[spinup:].reshape(res_seq[spinup:].shape[0], res_seq.shape[1], -1), beta)
    # print(model.readout.groups)
    # cmat = jnp.empty((model.readout.groups, int(model.data_dim / model.readout.groups), model.res_dim))
    # print(cmat.shape)
    # for jj in range(model.readout.groups):
    #     cmat_temp = solve_single_ridge_reg(res_seq[spinup:, jj, :], target_seq[spinup:, jj*int(model.data_dim / model.readout.groups): (jj+1)*int(model.data_dim / model.readout.groups)], 8e-8)
    #     cmat = cmat.at[jj].set(cmat_temp)

    # replace wout with learned weights
    def where(m):
        return m.readout.wout

    model = eqx.tree_at(where, model, cmat)
    return model, res_seq
