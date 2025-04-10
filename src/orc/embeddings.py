"""Define base class for embedding layers and implement common architectures."""

from abc import ABC, abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class EmbedBase(eqx.Module, ABC):
    """
    Base class dictating API for all implemented embedding layers.

    Attributes
    ----------
    in_dim : int
        Input dimension.
    res_dim : int
        Reservoir dimension.
    dtype : Float
        Dtype of JAX arrays, jnp.float32 or jnp.float64.

    Methods
    -------
    embed(in_state)
        Embed input into reservoir dimension.
    batch_embed(in_state)
        Embed multiple inputs into reservoir dimension.
    """

    in_dim: int
    res_dim: int
    dtype: Float

    def __init__(self, in_dim, res_dim, dtype=jnp.float64):
        """Ensure in dim, res dim,  and dtype are correct type."""
        self.res_dim = res_dim
        self.in_dim = in_dim
        self.dtype = dtype
        if not isinstance(res_dim, int):
            raise TypeError("Reservoir dimension res_dim must be an integer.")
        if not isinstance(in_dim, int):
            raise TypeError("Reservoir dimension res_dim must be an integer.")
        self.dtype = dtype
        if not (dtype == jnp.float64 or dtype == jnp.float32):
            raise TypeError("dtype must be jnp.float64 of jnp.float32.")

    @abstractmethod
    def embed(
        self,
        in_state: Array,
    ) -> Array:
        """Embed input signal to reservoir dimension.

        Parameters
        ----------
        in_state : Array
            Input state, (shape=(in_dim,)).

        Returns
        -------
        Array
            Embedded input state to reservoir dimension(shape=(res_dim,)).
        """
        pass

    def batch_embed(
        self,
        in_state: Array,
    ) -> Array:
        """Batch apply readout from reservoir states.

        Parameters
        ----------
        in_state : Array
            Input states, (shape=(batch_dim, in_dim,)).

        Returns
        -------
        Array
            Embedded input states to reservoir, (shape=(batch_dim, res_dim,)).
        """
        return eqx.filter_vmap(self.embed)(in_state)


class LinearEmbedding(EmbedBase):
    """Linear embedding layer.

    Attributes
    ----------
    in_dim : int
        Reservoir output dimension.
    res_dim : int
        Reservoir dimension.
    scaling : float
        Min/max values of input matrix.
    win : Array
        Input matrix.

    Methods
    -------
    embed(in_state)
        Embed input state to reservoir dimension.
    """

    in_dim: int
    res_dim: int
    scaling: float
    win: Array
    dtype: Float

    def __init__(
        self,
        in_dim: int,
        res_dim: int,
        scaling: float,
        dtype: Float = jnp.float64,
        *,
        seed: int,
    ) -> None:
        """Instantiate linear embedding.

        Parameters
        ----------
        in_dim : int
            Input dimension to reservoir.
        res_dim : int
            Reservoir dimension.
        scaling : float
            Min/max values of input matrix.
        seed : int
            Random seed for generating the PRNG key for the reservoir computer.
        dtype : Float
            Dtype of model, jnp.float64 or jnp.float32.
        """
        super().__init__(in_dim=in_dim, res_dim=res_dim, dtype=dtype)
        self.in_dim = in_dim
        self.res_dim = res_dim
        self.scaling = scaling
        self.dtype = dtype
        key = jax.random.key(seed)
        self.win = jax.random.uniform(
            key, (res_dim, in_dim), minval=-scaling, maxval=scaling, dtype=dtype
        )

    def embed(self, in_state: Array) -> Array:
        """Embed into reservoir dimension.

        Parameters
        ----------
        in_state : Array
            Input state, (shape=(in_dim,)).

        Returns
        -------
        Array
            Embedded input to reservoir, (shape=(res_dim,)).
        """
        if in_state.shape[0] != self.in_dim:
            raise ValueError(
                "Incorrect input dimension for instantiated embedding map."
            )
        return self.win @ in_state
