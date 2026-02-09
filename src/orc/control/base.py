"""Defines base classes for Reservoir Computer Controllers."""

from abc import ABC

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.optimize
from jaxtyping import Array, Float
import optax

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
    alpha_1 : float
        Weight for trajectory deviation penalty in control optimization.
    alpha_2 : float
        Weight for control magnitude penalty in control optimization.
    alpha_3 : float
        Weight for control derivative penalty in control optimization.
    seed : int
        Random seed for generating the PRNG key for the reservoir computer.


    Methods
    -------
    force(in_seq, control_seq, res_state)
        Teacher forces the reservoir with input and control sequences.
    apply_control(control_seq, res_state)
        Apply a predefined control sequence in closed-loop.
    set_readout(readout)
        Replaces the readout layer of the reservoir computer.
    set_embedding(embedding)
        Replaces the embedding layer of the reservoir computer.
    compute_penalty(control_seq, res_state, ref_traj)
        Compute the control penalty for a given control sequence.
    compute_control(control_seq, res_state, ref_traj)
        Compute optimal control sequence to track a reference trajectory.
    """

    driver: DriverBase
    readout: ReadoutBase
    embedding: EmbedBase
    in_dim: int
    control_dim: int
    out_dim: int
    res_dim: int
    dtype: Float = jnp.float64
    alpha_1: float = 100
    alpha_2: float = 1
    alpha_3: float = 5
    seed: int = 0

    def __init__(
        self,
        driver: DriverBase,
        readout: ReadoutBase,
        embedding: EmbedBase,
        in_dim: int,
        control_dim: int,
        dtype: Float = jnp.float64,
        alpha_1: float = 100,
        alpha_2: float = 1,
        alpha_3: float = 5,
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
        alpha_1 : float
            Weight for trajectory deviation penalty in control optimization.
        alpha_2 : float
            Weight for control magnitude penalty in control optimization.
        alpha_3 : float
            Weight for control derivative penalty in control optimization.
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
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_3 = alpha_3
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
        self, control_seq: Array, res_state: Array
    ) -> tuple[Array, Array]:
        """Apply a predefined control sequence in closed-loop.

        The readout feeds back as the next input: u(t+1) = readout(x(t)).
        Control c(t) comes from the provided control_seq.

        Parameters
        ----------
        control_seq : Array
            Control sequence to apply, (shape=(fcast_len, control_dim)).
        res_state : Array
            Initial reservoir state, (shape=(res_dim,)).

        Returns
        -------
        Array
            Controlled output trajectory with shape=(fcast_len, out_dim)).
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
        res_state, state_seq = jax.lax.scan(scan_fn, res_state, control_seq)
        return state_seq

    def set_readout(self, readout: ReadoutBase) -> "RCControllerBase":
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

    def set_embedding(self, embedding: EmbedBase) -> "RCControllerBase":
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
    
    def compute_penalty(
        self,
        control_seq: Array,
        res_state: Array,
        ref_traj: Array,
    ) -> Float:
        """Compute the control penalty for a given control sequence.

        The penalty consists of three terms:
        - Deviation penalty: squared error between forecast and reference trajectory
        - Magnitude penalty: squared norm of control inputs
        - Derivative penalty: squared norm of control input differences

        Parameters
        ----------
        control_seq : Array
            Control sequence to evaluate, (shape=(fcast_len, control_dim)).
        res_state : Array
            Initial reservoir state, (shape=(res_dim,)).
        ref_traj : Array
            Reference trajectory to track, (shape=(fcast_len, out_dim)).

        Returns
        -------
        Float
            Total penalty value (scalar).
        """
        fcast, _ = self.apply_control(control_seq, res_state)
        deviation = fcast - ref_traj
        dev_penalty = jnp.sum(deviation ** 2) * self.alpha_1
        mag_penalty = jnp.sum(control_seq ** 2) * self.alpha_2
        deriv_penalty = jnp.sum(jnp.diff(control_seq, axis=0) ** 2) * self.alpha_3
        return dev_penalty + mag_penalty + deriv_penalty
    
    @eqx.filter_jit
    def compute_control(
        self,
        control_seq: Array,
        res_state: Array,
        ref_traj: Array,
    ) -> Array:
        """Compute optimal control sequence to track a reference trajectory.

        Uses BFGS optimization to find a control sequence that minimizes the
        penalty function (deviation from reference + control magnitude + control
        smoothness).

        Parameters
        ----------
        control_seq : Array
            Initial guess for control sequence, (shape=(fcast_len, control_dim)).
        res_state : Array
            Initial reservoir state, (shape=(res_dim,)).
        ref_traj : Array
            Reference trajectory to track, (shape=(fcast_len, out_dim)).

        Returns
        -------
        Array
            Optimized control sequence, (shape=(fcast_len, control_dim)).
        """

        def loss_fn(control_seq):
            control_seq = control_seq.reshape(-1, self.control_dim)
            return self.compute_penalty(control_seq, res_state, ref_traj)

        # TODO: Implement optimization to allow finer grained control of latency tolerances
        # linesearch = optax.scale_by_backtracking_linesearch(
        #     max_backtracking_steps=30,
        #     decrease_factor=0.5,
        # )
        # solver = optax.lbfgs(linesearch=linesearch)

        # if not use_builtin_solver:
        #     solver = optax.lbfgs()

        #     @jax.jit
        #     def run_lbfgs(x0):
        #         value_and_grad_fn = jax.value_and_grad(loss_fn)

        #         def step(carry, _):
        #             x, state = carry
        #             value, grad = value_and_grad_fn(x)
        #             updates, state = solver.update(grad, state, x, value=value, grad=grad, value_fn=loss_fn)
        #             x = optax.apply_updates(x, updates)
        #             return (x, state), None

        #         init_state = solver.init(x0)
        #         (x_final, final_state), _ = jax.lax.scan(step, (x0, init_state), None, length=max_iter)
        #         return x_final, final_state

        #     control_opt, state = run_lbfgs(control_seq)

        optimize_results = jax.scipy.optimize.minimize(loss_fn, control_seq.reshape(-1), method='BFGS')
        control_opt = optimize_results.x.reshape(-1, self.control_dim)
        
        return control_opt
