"""Define base class for readout layers and implement common architectures."""

from abc import ABC, abstractmethod

import equinox as eqx
import jax
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
            Reservoir state, (shape=(res_dim,)).

        Returns
        -------
        Array
            Output from reservoir state, (shape=(out_dim,)).
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
            Reservoir state, (shape=(batch_dim, res_dim,)).

        Returns
        -------
        Array
            Output from reservoir states, (shape=(batch_dim, out_dim,)).
        """
        return eqx.filter_vmap(self.readout)(res_state)


class LinearReadout(ReadoutBase):
    """Linear readout layer.

    Attributes
    ----------
    out_dim : int
        Dimension of reservoir output.
    res_dim : int
        Reservoir dimension.
    groups : int
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
    groups: int
    dtype: Float

    def __init__(
        self,
        out_dim: int,
        res_dim: int,
        groups: int = 1,
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
        groups : int
            Number of parallel resrevoirs.
        dtype : Float
            Dtype, default jnp.float64.
        seed : int
            Not used for LinearReadout, present to maintain consistent interface.
        """
        super().__init__(out_dim=out_dim, res_dim=res_dim, dtype=dtype)
        self.out_dim = out_dim
        self.res_dim = res_dim
        self.wout = jnp.zeros((groups, out_dim, res_dim), dtype=dtype)
        self.dtype = dtype
        self.groups = groups

    @eqx.filter_jit
    def readout(self, res_state: Array) -> Array:
        """Readout from reservoir state.

        Parameters
        ----------
        res_state : Array
            Reservoir state, (shape=(groups, res_dim,)).

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