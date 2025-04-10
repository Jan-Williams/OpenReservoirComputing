"""Define base class for readout layers and implement common architectures."""

from abc import ABC, abstractmethod

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
    wout : Array
        output matrix
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
    dtype: Float

    def __init__(
        self,
        out_dim: int,
        res_dim: int,
        dtype: Float = jnp.float64,
        *,
        seed: int,
    ) -> None:
        """Initialize readout layer to zeros.

        Parameters
        ----------
        out_dim : int
            Dimension of reservoir output.
        res_dim : int
            Reservoir dimension.
        dtype : Float
            Dtype, default jnp.float64.
        seed : int
            Not used for LinearReadout, present to maintain consistent interface.
        """
        super().__init__(out_dim=out_dim, res_dim=res_dim, dtype=dtype)
        self.out_dim = out_dim
        self.res_dim = res_dim
        self.wout = jnp.zeros((out_dim, res_dim), dtype=dtype)
        self.dtype = dtype

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
        if res_state.shape[0] != self.res_dim:
            raise ValueError(
                "Incorrect reservoir dimension for instantiated output map."
            )
        return self.wout @ res_state
