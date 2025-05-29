"""Define base class for readout layers and implement common architectures."""

from abc import ABC, abstractmethod
from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class ReadoutBase(eqx.Module, ABC):
    """
    Base class dictating API for all implemented readout layers.

    Attributes
    ----------
    out_dim : int
        Dimension of reservoir output.
    res_dim : int
        Reservoir dimension.
    dtype : Float
        Dtype of JAX arrays, jnp.float32 or jnp.float64.

    Methods
    -------
    readout(res_state)
        Map from reservoir state to output state.
    batch_readout(res_state)
        Map from reservoir states to output states.
    """

    out_dim: int
    res_dim: int
    dtype: Float

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
        return self.readout(res_state)

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
        """Readout from reservoir state.

        If readout supports parallel reservoirs, this method needs to be overwritten
        to accomodate shape handling.

        Parameters
        ----------
        res_state : Array
            Reservoir state.

        Returns
        -------
        Array
            Output from reservoir state.
        """
        return self.readout(res_state)

class LinearReadout(ReadoutBase):
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
    dtype : Float
            Dtype, default jnp.float64.

    Methods
    -------
    readout(res_state)
        Map from reservoir state to output state.
    """

    out_dim: int
    res_dim: int
    wout: Array
    chunks: int
    dtype: Float

    def __init__(
        self,
        out_dim: int,
        res_dim: int,
        chunks: int = 1,
        dtype: Float = jnp.float64,
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
        dtype : Float
            Dtype, default jnp.float64.
        seed : int
            Not used for LinearReadout, present to maintain consistent interface.
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

    def __call__(self, res_state: Array) -> Array:
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
        if len(res_state.shape) == 2:
            to_ret = self.readout(res_state)
        elif len(res_state.shape) == 3:
            to_ret = self.batch_readout(res_state)
        else:
            raise ValueError(
                "Only 1-dimensional localization is currently supported, detected a "
                f"{len(res_state.shape)}D field."
            )
        return to_ret


class NonlinearReadout(ReadoutBase):
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
    dtype : Float
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

    out_dim: int
    res_dim: int
    wout: Array
    chunks: int
    nonlin_list: list
    dtype: Float

    def __init__(
        self,
        out_dim: int,
        res_dim: int,
        nonlin_list: list[Callable],
        chunks: int = 1,
        dtype: Float = jnp.float64,
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
        dtype : Float
            Dtype, default jnp.float64.
        seed : int
            Not used for NonlinearReadout, present to maintain consistent interface.
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
            Reservoir state, (shape=(chunks, res_dim,)).

        Returns
        -------
        Array
            Transformed reservoir state.
        """
        num_nonlins = len(self.nonlin_list)
        for idx in range(num_nonlins):
            transformed_res_state = res_state.at[:, idx + 1 :: num_nonlins + 1].set(
                self.nonlin_list[idx](res_state[:, idx + 1 :: num_nonlins + 1])
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

    def __call__(self, res_state: Array) -> Array:
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
        if len(res_state.shape) == 2:
            to_ret = self.readout(res_state)
        elif len(res_state.shape) == 3:
            to_ret = self.batch_readout(res_state)
        else:
            raise ValueError(
                "Only 1-dimensional localization is currently supported, detected a "
                f"{len(res_state.shape)}D field."
            )
        return to_ret


class QuadraticReadout(NonlinearReadout):
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
    dtype : Float
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

    out_dim: int
    res_dim: int
    wout: Array
    chunks: int
    dtype: Float

    def __init__(
        self,
        out_dim: int,
        res_dim: int,
        chunks: int = 1,
        dtype: Float = jnp.float64,
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
        dtype : Float
            Dtype, default jnp.float64.
        seed : int
            Not used for QuadraticReadout, present to maintain consistent interface.
        """
        super().__init__(
            out_dim=out_dim,
            res_dim=res_dim,
            dtype=dtype,
            nonlin_list=[lambda x: x ** 2],
            chunks=chunks,
        )
