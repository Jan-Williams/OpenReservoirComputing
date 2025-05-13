"""Define base class for Reservoir Computers."""

from abc import ABC

import diffrax
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
    chunks: int = 1
    seed: int = 0

    @eqx.filter_jit
    def force(self, in_seq: Array, res_state: Array) -> Array:
        """Teacher forces the reservoir.

        Parameters
        ----------
        in_seq: Array
            Input sequence to force the reservoir, (shape=(seq_len, data_dim)).
        res_state : Array
            Initial reservoir state, (shape=(chunks, res_dim,)).

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

    @eqx.filter_jit
    def forecast_from_IC(self, fcast_len: int, spinup_data: Array) -> Array:
        """Forecast from a sequence of spinup data.

        Parameters
        ----------
        fcast_len : int
            Steps to forecast.
        spinup_data : Array
            Initial condition sequence, (shape=(seq_len, data_dim)).

        Returns
        -------
        Array
            Forecasted states, (shape=(fcast_len, data_dim)).
        """
        res_seq = self.force(
            spinup_data, jnp.zeros((self.chunks, self.res_dim), dtype=self.dtype)
        )
        return self.forecast(fcast_len, res_seq[-1])

class CRCForecasterBase(RCForecasterBase, ABC):
    """Base class for continuous reservoir computer forecasters.

    Override the force and forecast methods of RCForecasterBase
    to timestep the RC forward using a continuous time ODE solver.

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
    solver : diffrax.Solver
        ODE solver to use for the reservoir computer.
    stepsize_controller : diffrax.StepsizeController
        Stepsize controller to use for the ODE solver.

    Methods
    -------
    force(in_seq, res_state)
        Teacher forces the reservoir with the input sequence.
    set_readout(readout)
        Replaces the readout layer of the reservoir computer.
    set_embedding(embedding)
        Replaces the embedding layer of the reservoir computer.
    """

    solver: diffrax.AbstractSolver
    stepsize_controller: diffrax.AbstractAdaptiveStepSizeController

    def __init__(self,
                *args,
                solver: diffrax.AbstractSolver = None,
                stepsize_controller:
                    diffrax.AbstractAdaptiveStepSizeController = None,
                **kwargs):
        """Initialize the continuous reservoir computer.

        Parameters
        ----------
        *args : tuple
            Positional arguments to pass to the parent class RCForecasterBase.
        solver : diffrax.AbstractSolver
            ODE solver to use for the reservoir computer.
        stepsize_controller : diffrax.AbstractAdaptiveStepSizeController
            Stepsize controller to use for the ODE solver.
        **kwargs : dict
            Keyword arguments to pass to the parent class RCForecasterBase.
        """
        super().__init__(*args, **kwargs)
        if solver is None:
            solver = diffrax.Tsit5()
        if stepsize_controller is None:
            stepsize_controller = diffrax.PIDController(rtol=1e-3,
                                                        atol=1e-6,
                                                        icoeff=1.0)
        self.solver = solver
        self.stepsize_controller = stepsize_controller

    def force(self, in_seq: Array, res_state: Array, ts: Array) -> Array:
        """
        Teacher forces the reservoir.

        Parameters
        ----------
        in_seq: Array
            Input sequence to force the reservoir, (shape=(seq_len, data_dim)).
        res_state : Array
            Initial reservoir state, (shape=(res_dim,)).
        ts: Array
            Time steps for the input sequence, (shape=(seq_len,)).

        Returns
        -------
        Array
            Forced reservoir sequence, (shape=(seq_len, res_dim)).
        """
        # form interpolants
        coeffs = diffrax.backward_hermite_coefficients(ts, in_seq)
        in_seq_interp = diffrax.CubicInterpolation(ts, coeffs)

        # RC forced ODE definition
        @eqx.filter_jit
        def res_ode(t,r,args):
            interp = args
            proj_vars = self.embedding(interp.evaluate(t))
            return self.driver(proj_vars, r)

        # integrate RC
        dt0 = ts[1] - ts[0]
        ts = ts + dt0 # roll time forward one step for targets
        term = diffrax.ODETerm(res_ode)
        args = in_seq_interp
        save_at = diffrax.SaveAt(ts=ts)
        sol = diffrax.diffeqsolve(term,
                                    t0=0.0,
                                    t1=ts[-1],
                                    dt0=dt0,
                                    y0=res_state,
                                    solver=self.solver,
                                    stepsize_controller=self.stepsize_controller,
                                    args=args,
                                    saveat=save_at,
                                    max_steps=None)
        res_seq = sol.ys
        return res_seq

    def forecast(self, ts: Array, res_state: Array) -> Array:
        """Forecast from an initial reservoir state.

        Parameters
        ----------
        ts : Array
            Time steps for the forecast, (shape=(fcast_len,)).
        res_state : Array
            Initial reservoir state, (shape=(res_dim)).

        Returns
        -------
        Array
            Forecasted states, (shape=(fcast_len, data_dim))
        """
        # RC autonomous ODE definition
        @eqx.filter_jit
        def res_ode(t, r, args):
            out_state = self.driver(self.embedding(self.readout(r)), r)
            return out_state

        # integrate RC
        dt0 = ts[1] - ts[0]
        ts = ts+dt0
        term = diffrax.ODETerm(res_ode)
        save_at = diffrax.SaveAt(ts=ts)
        sol = diffrax.diffeqsolve(term,
                                    t0=0.0,
                                    t1=ts[-1],
                                    dt0=dt0,
                                    y0=res_state,
                                    solver=self.solver,
                                    stepsize_controller=self.stepsize_controller,
                                    saveat=save_at,
                                    max_steps=None)
        res_seq = sol.ys
        return eqx.filter_vmap(self.readout)(res_seq)

    def forecast_from_IC(self, ts:Array,
                         spinup_data: Array,
                         spinup_ts: Array = None) -> Array:
        """Forecast from a sequence of spinup data.

        Parameters
        ----------
        ts : Array
            Time steps for the forecast, (shape=(fcast_len,)).
        spinup_data : Array
            Initial condition sequence, (shape=(seq_len, data_dim)).
        spinup_ts : Array
            Time steps for the spinup data, (shape=(seq_len,)).
            If None, the spinup data is assumed to have the same dt
            as the forecast data.  If not None, the spinup data
            Default is None.

        Returns
        -------
        Array
            Forecasted states, (shape=(fcast_len, data_dim)).
        """
        if spinup_ts is None:
            dt0 = ts[1] - ts[0]
            spinup_ts = jnp.arange(
                0.0, spinup_data.shape[0], dtype=self.dtype
            ) * dt0

        res_seq = self.force(
            spinup_data,
            jnp.zeros((self.chunks, self.res_dim), dtype=self.dtype),
            spinup_ts
        )

        return self.forecast(ts, res_seq[-1])
