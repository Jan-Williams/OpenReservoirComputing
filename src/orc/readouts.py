"""Define base class for readout layers and implement common architectures."""

from abc import ABC, abstractmethod
from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array


class ReadoutBase(eqx.Module, ABC):
    """
    Base class dictating API for all implemented readout layers.

    All readout layers should store their trained weight matrix as ``wout``.

    Attributes
    ----------
    out_dim : int
        Dimension of reservoir output.
    res_dim : int
        Reservoir dimension.
    chunks : int
        Number of parallel reservoirs. Default is 0 (no chunks dimension).
    dtype : type
        Dtype of JAX arrays, jnp.float32 or jnp.float64.

    Methods
    -------
    readout(res_state)
        Map from reservoir state to output state.
    batch_readout(res_state)
        Map from reservoir states to output states.
    prepare_train(res_seq)
        Prepare reservoir states for ridge regression during training.
    prepare_target(target_seq)
        Prepare target sequence for ridge regression during training.
    set_wout(cmat)
        Return a new readout with updated output weights.
    """

    out_dim: int = eqx.field(static=True)
    res_dim: int = eqx.field(static=True)
    chunks: int = eqx.field(default=0, static=True)
    dtype: type = eqx.field(default=jnp.float64, static=True)

    def __init__(self, out_dim, res_dim, dtype=jnp.float64):
        """Ensure in dim, res dim, and dtype are correct type."""
        self.res_dim = res_dim
        self.out_dim = out_dim
        self.dtype = dtype
        if not isinstance(res_dim, int):
            raise TypeError("Reservoir dimension res_dim must be an integer.")
        if not isinstance(out_dim, int):
            raise TypeError("Reservoir dimension res_dim must be an integer.")
        self.dtype = dtype
        if not (dtype == jnp.float64 or dtype == jnp.float32):
            raise TypeError("dtype must be jnp.float64 of jnp.float32.")

    @abstractmethod
    def readout(
        self,
        res_state: Array,
    ) -> Array:
        """Readout from reservoir state.

        Parameters
        ----------
        res_state : Array
            Reservoir state.

        Returns
        -------
        Array
            Output from reservoir state.
        """
        pass

    def batch_readout(
        self,
        res_state: Array,
    ) -> Array:
        """Batch apply readout from reservoir states.

        Parameters
        ----------
        res_state : Array
            Reservoir state.

        Returns
        -------
        Array
            Output from reservoir states.
        """
        return eqx.filter_vmap(self.readout)(res_state)

    def __call__(
        self,
        res_state: Array,
    ) -> Array:
        """Call either readout or batch_readout depending on dimensions.

        Parameters
        ----------
        res_state : Array
            Reservoir state, (shape=(chunks, res_dim) or
            shape=(seq_len, chunks, res_dim)).

        Returns
        -------
        Array
            Output state, (out_dim,) or shape=(seq_len, out_dim)).
        """
        if self.chunks > 0:
            if len(res_state.shape) == 2:
                to_ret = self.readout(res_state)
            elif len(res_state.shape) == 3:
                to_ret = self.batch_readout(res_state)
        else:
            if len(res_state.shape) == 1:
                to_ret = self.readout(res_state)
            elif len(res_state.shape) == 2:
                to_ret = self.batch_readout(res_state)
        return to_ret

    def prepare_train(self, res_seq: Array) -> Array:
        """Prepare reservoir states for ridge regression during training.

        By default, returns the reservoir states unchanged. Subclasses may
        override to apply nonlinear transforms or reshape for chunk structure.

        Parameters
        ----------
        res_seq : Array
            Sequence of reservoir states.

        Returns
        -------
        Array
            Prepared reservoir states for ridge regression.
        """
        return res_seq

    def prepare_target(self, target_seq: Array) -> Array:
        """Prepare target sequence for ridge regression during training.

        By default, returns the target sequence unchanged. Subclasses may
        override to reshape for chunk structure.

        Parameters
        ----------
        target_seq : Array
            Target sequence, (shape=(seq_len, out_dim)).

        Returns
        -------
        Array
            Prepared target sequence for ridge regression.
        """
        return target_seq

    def set_wout(self, cmat: Array):
        """Return a new readout with updated output weights.

        Override this method if your readout uses a different
        attribute name for the output weight matrix.

        Parameters
        ----------
        cmat : Array
            New output weight matrix.

        Returns
        -------
        ReadoutBase
            New readout with updated weights.
        """
        return eqx.tree_at(lambda r: r.wout, self, cmat)


class ParallelLinearReadout(ReadoutBase):
    """Linear readout layer.

    Attributes
    ----------
    out_dim : int
        Dimension of reservoir output.
    res_dim : int
        Reservoir dimension.
    chunks : int
        Number of parallel reservoirs.
    wout : Array
        Output matrix.
    dtype : type
            Dtype, default jnp.float64.

    Methods
    -------
    readout(res_state)
        Map from reservoir state to output state.
    """

    out_dim: int = eqx.field(static=True)
    res_dim: int = eqx.field(static=True)
    wout: Array
    chunks: int = eqx.field(static=True)
    dtype: type = eqx.field(static=True)

    def __init__(
        self,
        out_dim: int,
        res_dim: int,
        chunks: int = 1,
        dtype: type = jnp.float64,
        *,
        seed: int = 0,
    ) -> None:
        """Initialize readout layer to zeros.

        Parameters
        ----------
        out_dim : int
            Dimension of reservoir output.
        res_dim : int
            Reservoir dimension.
        chunks : int
            Number of parallel resrevoirs.
        dtype : type
            Dtype, default jnp.float64.
        seed : int
            Not used for ParallelLinearReadout, here to maintain consistent
            interface.
        """
        super().__init__(out_dim=out_dim, res_dim=res_dim, dtype=dtype)
        self.out_dim = out_dim
        self.res_dim = res_dim
        self.wout = jnp.zeros((chunks, int(out_dim / chunks), res_dim), dtype=dtype)
        self.dtype = dtype
        self.chunks = chunks

    @eqx.filter_jit
    def readout(self, res_state: Array) -> Array:
        """Readout from reservoir state.

        Parameters
        ----------
        res_state : Array
            Reservoir state, (shape=(chunks, res_dim,)).

        Returns
        -------
        Array
            Output from reservoir, (shape=(out_dim,)).
        """
        if res_state.shape[1] != self.res_dim:
            raise ValueError(
                "Incorrect reservoir dimension for instantiated output map."
            )
        return jnp.ravel(eqx.filter_vmap(jnp.matmul)(self.wout, res_state))

    def prepare_target(self, target_seq: Array) -> Array:
        """Reshape target for parallel chunk structure.

        Parameters
        ----------
        target_seq : Array
            Target sequence, (shape=(seq_len, out_dim)).

        Returns
        -------
        Array
            Reshaped target, (shape=(seq_len, chunks, out_dim/chunks)).
        """
        return target_seq.reshape(target_seq.shape[0], self.chunks, -1)


class LinearReadout(ParallelLinearReadout):
    """Linear readout layer.

    Attributes
    ----------
    out_dim : int
        Dimension of reservoir output.
    res_dim : int
        Reservoir dimension.
    wout : Array
        Output matrix.
    dtype : type
            Dtype, default jnp.float64.

    Methods
    -------
    readout(res_state)
        Map from reservoir state to output state.
    """

    def __init__(
        self,
        out_dim: int,
        res_dim: int,
        dtype: type = jnp.float64,
        *,
        seed: int = 0,
    ) -> None:
        """Initialize readout layer to zeros.

        Parameters
        ----------
        out_dim : int
            Dimension of reservoir output.
        res_dim : int
            Reservoir dimension.
        dtype : type
            Dtype, default jnp.float64.
        seed : int
            Not used for ParallelLinearReadout, here to maintain consistent
            interface.
        """
        super().__init__(
            out_dim=out_dim,
            res_dim=res_dim,
            chunks=1,
            dtype=dtype,
            seed=seed,
        )

    @eqx.filter_jit
    def readout(self, res_state: Array) -> Array:
        """Readout from reservoir state.

        Parameters
        ----------
        res_state : Array
            Reservoir state, (shape=(res_dim,)).

        Returns
        -------
        Array
            Output from reservoir, (shape=(out_dim,)).
        """
        return super().readout(res_state.reshape(1, -1)).reshape(-1)

    def __call__(self, res_state: Array) -> Array:
        """Call either readout or batch_readout depending on dimensions.

        Parameters
        ----------
        res_state : Array
            Reservoir state, (shape=(res_dim) or
            shape=(seq_len, res_dim)).

        Returns
        -------
        Array
            Output state, (out_dim,) or shape=(seq_len, out_dim)).
        """
        return jnp.squeeze(super().__call__(res_state[..., None, :]))

    def prepare_train(self, res_seq: Array) -> Array:
        """Unsqueeze to add chunks=1 axis for ridge regression.

        Parameters
        ----------
        res_seq : Array
            Reservoir states, (shape=(seq_len, res_dim)).

        Returns
        -------
        Array
            Unsqueezed states, (shape=(seq_len, 1, res_dim)).
        """
        return res_seq[:, None, :]

    def prepare_target(self, target_seq: Array) -> Array:
        """Unsqueeze to add chunks=1 axis for ridge regression.

        Parameters
        ----------
        target_seq : Array
            Target sequence, (shape=(seq_len, out_dim)).

        Returns
        -------
        Array
            Unsqueezed target, (shape=(seq_len, 1, out_dim)).
        """
        return target_seq[:, None, :]


class ParallelNonlinearReadout(ReadoutBase):
    """Readout layer with user specified nonlinearities.

    Attributes
    ----------
    out_dim : int
        Dimension of reservoir output.
    res_dim : int
        Reservoir dimension.
    chunks : int
        Number of parallel reservoirs.
    wout : Array
        Output matrix.
    nonlin_list : list
        List containing user specified nonlinearities.
    dtype : type
            Dtype, default jnp.float64.

    Methods
    -------
    nonlinear_transform(res_state)
        Nonlinear transform that acts entrywise on reservoir state.
    readout(res_state)
        Map from reservoir state to output state.
    __call__(res_state)
        Map from reservoir state to output state, handles batch and single outputs.
    """

    out_dim: int = eqx.field(static=True)
    res_dim: int = eqx.field(static=True)
    wout: Array
    chunks: int = eqx.field(static=True)
    nonlin_list: list
    dtype: type = eqx.field(static=True)

    def __init__(
        self,
        out_dim: int,
        res_dim: int,
        nonlin_list: list[Callable],
        chunks: int = 1,
        dtype: type = jnp.float64,
        *,
        seed: int = 0,
    ) -> None:
        """Initialize readout layer to zeros.

        Parameters
        ----------
        out_dim : int
            Dimension of reservoir output.
        res_dim : int
            Reservoir dimension.
        nonlin_list : list[Callable]
            List containing user specified entrywise nonlinearities. Each entry should
            be a function mapping a scalar value to another scalar value, e.g.
            lambda x : x ** 2 or lambda x : jnp.sin(x).
        chunks : int
            Number of parallel reservoirs.
        dtype : type
            Dtype, default jnp.float64.
        seed : int
            Not used for ParallelNonlinearReadout, here to maintain consistent
            interface.
        """
        super().__init__(out_dim=out_dim, res_dim=res_dim, dtype=dtype)
        self.out_dim = out_dim
        self.res_dim = res_dim
        self.wout = jnp.zeros((chunks, int(out_dim / chunks), res_dim), dtype=dtype)
        self.dtype = dtype
        self.chunks = chunks
        self.nonlin_list = nonlin_list

    def nonlinear_transform(self, res_state: Array) -> Array:
        """Perform nonlinear transformation on reservoir state.

        Let tot_list be the list consisting of nonlin_list prepended by the identity
        mapping. Let n be the length of tot_list. Then, nonlinear_transform acts such
        that for all 0 <= k < chunks and 0 <= j < j * n:
        res_state[k, j * n] <- res_state[k, j*n]
        res_state[k, j * n + 1] <- f_0(res_state[k, j * n + 1])
        ...
        res_state[k, j * n + n - 1] <- f_{n-1}(res_state[k, j * n + n - 1])
        where f_i is the i-th entry of nonlin_list.

        Parameters
        ----------
        res_state : Array
            Reservoir state, (shape=(..., res_dim,)).

        Returns
        -------
        Array
            Transformed reservoir state.
        """
        num_nonlins = len(self.nonlin_list)
        transformed_res_state = res_state
        for idx in range(num_nonlins):
            transformed_res_state = transformed_res_state.at[
                ..., idx + 1 :: num_nonlins + 1
            ].set(
                self.nonlin_list[idx](
                    transformed_res_state[..., idx + 1 :: num_nonlins + 1]
                )
            )
        return transformed_res_state

    @eqx.filter_jit
    def readout(self, res_state: Array) -> Array:
        """Readout from reservoir state.

        Parameters
        ----------
        res_state : Array
            Reservoir state, (shape=(chunks, res_dim,)).

        Returns
        -------
        Array
            Output from reservoir, (shape=(out_dim,)).
        """
        if res_state.shape[1] != self.res_dim:
            raise ValueError(
                "Incorrect reservoir dimension for instantiated output map."
            )
        transformed_res_state = self.nonlinear_transform(res_state)
        return jnp.ravel(eqx.filter_vmap(jnp.matmul)(self.wout, transformed_res_state))

    def prepare_train(self, res_seq: Array) -> Array:
        """Apply nonlinear transform to reservoir states for ridge regression.

        Parameters
        ----------
        res_seq : Array
            Reservoir states, (shape=(seq_len, chunks, res_dim)).

        Returns
        -------
        Array
            Transformed states, (shape=(seq_len, chunks, res_dim)).
        """
        return self.nonlinear_transform(res_seq)

    def prepare_target(self, target_seq: Array) -> Array:
        """Reshape target for parallel chunk structure.

        Parameters
        ----------
        target_seq : Array
            Target sequence, (shape=(seq_len, out_dim)).

        Returns
        -------
        Array
            Reshaped target, (shape=(seq_len, chunks, out_dim/chunks)).
        """
        return target_seq.reshape(target_seq.shape[0], self.chunks, -1)


class NonlinearReadout(ParallelNonlinearReadout):
    """Readout layer with user specified nonlinearities.

    Attributes
    ----------
    out_dim : int
        Dimension of reservoir output.
    res_dim : int
        Reservoir dimension.
    chunks : int
        Number of parallel reservoirs.
    wout : Array
        Output matrix.
    nonlin_list : list
        List containing user specified nonlinearities.
    dtype : type
            Dtype, default jnp.float64.

    Methods
    -------
    nonlinear_transform(res_state)
        Nonlinear transform that acts entrywise on reservoir state.
    readout(res_state)
        Map from reservoir state to output state.
    __call__(res_state)
        Map from reservoir state to output state, handles batch and single outputs.
    """

    def __init__(
        self,
        out_dim: int,
        res_dim: int,
        nonlin_list: list[Callable],
        dtype: type = jnp.float64,
        *,
        seed: int = 0,
    ) -> None:
        """Initialize readout layer to zeros.

        Parameters
        ----------
        out_dim : int
            Dimension of reservoir output.
        res_dim : int
            Reservoir dimension.
        nonlin_list : list[Callable]
            List containing user specified entrywise nonlinearities. Each entry should
            be a function mapping a scalar value to another scalar value, e.g.
            lambda x : x ** 2 or lambda x : jnp.sin(x).
        dtype : type
            Dtype, default jnp.float64.
        seed : int
            Not used for ParallelNonlinearReadout, here to maintain consistent
            interface.
        """
        super().__init__(
            out_dim=out_dim,
            res_dim=res_dim,
            nonlin_list=nonlin_list,
            chunks=1,
            dtype=dtype,
            seed=seed,
        )

    def nonlinear_transform(self, res_state: Array) -> Array:
        """Perform nonlinear transformation on reservoir state.

        Let tot_list be the list consisting of nonlin_list prepended by the identity
        mapping. Let n be the length of tot_list. Then, nonlinear_transform acts such
        that for all 0 <= k < chunks and 0 <= j < j * n:
        res_state[k, j * n] <- res_state[k, j*n]
        res_state[k, j * n + 1] <- f_0(res_state[k, j * n + 1])
        ...
        res_state[k, j * n + n - 1] <- f_{n-1}(res_state[k, j * n + n - 1])
        where f_i is the i-th entry of nonlin_list.

        Parameters
        ----------
        res_state : Array
            Reservoir state, (shape=(..., res_dim,)).

        Returns
        -------
        Array
            Transformed reservoir state.
        """
        num_nonlins = len(self.nonlin_list)
        transformed_res_state = res_state
        for idx in range(num_nonlins):
            transformed_res_state = transformed_res_state.at[
                ..., idx + 1 :: num_nonlins + 1
            ].set(
                self.nonlin_list[idx](
                    transformed_res_state[..., idx + 1 :: num_nonlins + 1]
                )
            )
        return transformed_res_state

    @eqx.filter_jit
    def readout(self, res_state: Array) -> Array:
        """Readout from reservoir state.

        Parameters
        ----------
        res_state : Array
            Reservoir state, (shape=(res_dim,)).

        Returns
        -------
        Array
            Output from reservoir, (shape=(out_dim,)).
        """
        return super().readout(res_state.reshape(1, -1)).reshape(-1)

    def __call__(self, res_state: Array) -> Array:
        """Call either readout or batch_readout depending on dimensions.

        Parameters
        ----------
        res_state : Array
            Reservoir state, (shape=(res_dim) or
            shape=(seq_len, res_dim)).

        Returns
        -------
        Array
            Output state, (out_dim,) or shape=(seq_len, out_dim)).
        """
        return jnp.squeeze(super().__call__(res_state[..., None, :]))

    def prepare_train(self, res_seq: Array) -> Array:
        """Apply nonlinear transform and unsqueeze to add chunks=1 axis.

        Parameters
        ----------
        res_seq : Array
            Reservoir states, (shape=(seq_len, res_dim)).

        Returns
        -------
        Array
            Transformed and unsqueezed states, (shape=(seq_len, 1, res_dim)).
        """
        return self.nonlinear_transform(res_seq)[:, None, :]

    def prepare_target(self, target_seq: Array) -> Array:
        """Unsqueeze to add chunks=1 axis for ridge regression.

        Parameters
        ----------
        target_seq : Array
            Target sequence, (shape=(seq_len, out_dim)).

        Returns
        -------
        Array
            Unsqueezed target, (shape=(seq_len, 1, out_dim)).
        """
        return target_seq[:, None, :]


class ParallelQuadraticReadout(ParallelNonlinearReadout):
    """Quadratic readout layer.

    Attributes
    ----------
    out_dim : int
        Dimension of reservoir output.
    res_dim : int
        Reservoir dimension.
    chunks : int
        Number of parallel reservoirs.
    wout : Array
        Output matrix.
    dtype : type
            Dtype, default jnp.float64.

    Methods
    -------
    nonlinear_transform(res_state)
        Quadratic transform that acts entrywise on reservoir state.
    readout(res_state)
        Map from reservoir state to output state with quadratic nonlinearity.
    __call__(res_state)
        Map from reservoir state to output state with quadratic nonlinearity,
        handles batch and single outputs.
    """

    out_dim: int = eqx.field(static=True)
    res_dim: int = eqx.field(static=True)
    wout: Array
    chunks: int = eqx.field(static=True)
    dtype: type = eqx.field(static=True)

    def __init__(
        self,
        out_dim: int,
        res_dim: int,
        chunks: int = 1,
        dtype: type = jnp.float64,
        *,
        seed: int = 0,
    ) -> None:
        """Initialize readout layer to zeros.

        Parameters
        ----------
        out_dim : int
            Dimension of reservoir output.
        res_dim : int
            Reservoir dimension.
        chunks : int
            Number of parallel resrevoirs.
        dtype : type
            Dtype, default jnp.float64.
        seed : int
            Not used for ParallelQuadraticReadout, here to maintain consistent
            interface.
        """
        super().__init__(
            out_dim=out_dim,
            res_dim=res_dim,
            dtype=dtype,
            nonlin_list=[lambda x: x**2],
            chunks=chunks,
        )


class QuadraticReadout(NonlinearReadout):
    """Quadratic readout layer.

    Attributes
    ----------
    out_dim : int
        Dimension of reservoir output.
    res_dim : int
        Reservoir dimension.
    wout : Array
        Output matrix.
    dtype : type
            Dtype, default jnp.float64.

    Methods
    -------
    nonlinear_transform(res_state)
        Quadratic transform that acts entrywise on reservoir state.
    readout(res_state)
        Map from reservoir state to output state with quadratic nonlinearity.
    __call__(res_state)
        Map from reservoir state to output state with quadratic nonlinearity,
        handles batch and single outputs.
    """

    def __init__(
        self,
        out_dim: int,
        res_dim: int,
        dtype: type = jnp.float64,
        *,
        seed: int = 0,
    ) -> None:
        """Initialize readout layer to zeros.

        Parameters
        ----------
        out_dim : int
            Dimension of reservoir output.
        res_dim : int
            Reservoir dimension.
        dtype : type
            Dtype, default jnp.float64.
        seed : int
            Not used for ParallelQuadraticReadout, here to maintain consistent
            interface.
        """
        super().__init__(
            out_dim=out_dim,
            res_dim=res_dim,
            dtype=dtype,
            nonlin_list=[lambda x: x**2],
        )


class EnsembleLinearReadout(ReadoutBase):
    """Esembled linear readout layer.

    Attributes
    ----------
    out_dim : int
        Dimension of reservoir output.
    res_dim : int
        Reservoir dimension.
    chunks : int
        Number of parallel reservoirs.
    wout : Array
        Output matrix.
    dtype : type
            Dtype, default jnp.float64.

    Methods
    -------
    readout(res_state)
        Map from reservoir state to output state.
    """

    out_dim: int = eqx.field(static=True)
    res_dim: int = eqx.field(static=True)
    wout: Array
    chunks: int = eqx.field(static=True)
    dtype: type = eqx.field(static=True)

    def __init__(
        self,
        out_dim: int,
        res_dim: int,
        chunks: int = 5,
        dtype: type = jnp.float64,
        *,
        seed: int = 0,
    ) -> None:
        """Initialize readout layer to zeros.

        Parameters
        ----------
        out_dim : int
            Dimension of reservoir output.
        res_dim : int
            Reservoir dimension.
        chunks : int
            Number of parallel resrevoirs.
        dtype : type
            Dtype, default jnp.float64.
        seed : int
            Not used for ParallelLinearReadout, here to maintain consistent
            interface.
        """
        super().__init__(out_dim=out_dim, res_dim=res_dim, dtype=dtype)
        self.out_dim = out_dim
        self.res_dim = res_dim
        self.wout = jnp.zeros((chunks, out_dim, res_dim), dtype=dtype)
        self.dtype = dtype
        self.chunks = chunks

    @eqx.filter_jit
    def readout(self, res_state: Array) -> Array:
        """Readout from reservoir state.

        Parameters
        ----------
        res_state : Array
            Reservoir state, (shape=(chunks, res_dim,)).

        Returns
        -------
        Array
            Output from reservoir, (shape=(out_dim,)).
        """
        if res_state.shape[1] != self.res_dim:
            raise ValueError(
                "Incorrect reservoir dimension for instantiated output map."
            )
        return jnp.mean(eqx.filter_vmap(jnp.matmul)(self.wout, res_state), axis=0)

    def prepare_target(self, target_seq: Array) -> Array:
        """Repeat target across all ensemble members.

        Parameters
        ----------
        target_seq : Array
            Target sequence, (shape=(seq_len, out_dim)).

        Returns
        -------
        Array
            Repeated target, (shape=(seq_len, chunks, out_dim)).
        """
        return jnp.repeat(target_seq[:, None, :], self.chunks, axis=1)
