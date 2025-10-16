"""Define base class for embedding layers and implement common architectures."""

from abc import ABC, abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

jax.config.update("jax_enable_x64", True)


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
            Embedded input state to reservoir dimension.
        """
        pass

    @eqx.filter_jit
    def batch_embed(
        self,
        in_state: Array,
    ) -> Array:
        """Batch apply embedding from input states.

        Parameters
        ----------
        in_state : Array
            Input states.

        Returns
        -------
        Array
            Embedded input states to reservoir, (shape=(batch_dim, res_dim,)).
        """
        return eqx.filter_vmap(self.embed)(in_state)

    def __call__(
        self,
        in_state: Array,
    ) -> Array:
        """Embed input signal to reservoir dimension.

        If embedding supports parallel reservoirs, this method needs to be overwritten
        to accomodate shape handling.

        Parameters
        ----------
        in_state : Array
            Input state, (shape=(in_dim,)).

        Returns
        -------
        Array
            Embedded input state to reservoir dimension.
        """
        return self.embed(in_state)


class ParallelLinearEmbedding(EmbedBase):
    """Linear embedding layer.

    Attributes
    ----------
    in_dim : int
        Reservoir input dimension.
    res_dim : int
        Reservoir dimension.
    scaling : float
        Min/max values of input matrix.
    win : Array
        Input matrix.
    chunks : int
        Number of parallel reservoirs.
    locality : int
        Adjacent reservoir overlap.
    periodic : bool
        Assume periodic BCs when decomposing the input state to parallel network
        inputs. If False, the input is padded with boundary values at the edges
        (i.e., edge values are extended to the locality region), which may not
        match the true spatial dynamics. If True, the input is padded by connecting
        smoothly the end and beginning of the signal ensuring continuity. Default is
        True.

    Methods
    -------
    __call__(in_state)
        Embed input state to reservoir dimension.
    localize(in_state, periodic=True)
        Decompose input_state to parallel network inputs.
    moving_window(a)
        Helper function for localize.
    embed(in_state)
        Embed single input state to reservoir dimension.
    """

    in_dim: int
    res_dim: int
    scaling: float
    win: Array
    dtype: Float
    chunks: int
    locality: int
    chunk_size: int
    periodic: bool

    def __init__(
        self,
        in_dim: int,
        res_dim: int,
        scaling: float,
        dtype: Float = jnp.float64,
        chunks: int = 1,
        locality: int = 0,
        periodic: bool = True,
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
        periodic : bool
            Assume periodic BCs when decomposing the input state to parallel network
            inputs. If False, the input is padded with boundary values at the edges
            (i.e., edge values are extended to the locality region), which may not
            match the true spatial dynamics. If True, the input is padded by connecting
            smoothly the end and beginning of the signal ensuring continuity. Default is
            True.
        """
        super().__init__(in_dim=in_dim, res_dim=res_dim, dtype=dtype)
        self.scaling = scaling
        self.dtype = dtype
        key = jax.random.key(seed)
        self.chunk_size = int(in_dim / chunks)

        if in_dim % chunks:
            raise ValueError(
                f"The number of chunks {chunks} must evenly divide in_dim {in_dim}."
            )

        self.win = jax.random.uniform(
            key,
            (chunks, res_dim, self.chunk_size + 2 * locality),
            minval=-scaling,
            maxval=scaling,
            dtype=dtype,
        )
        self.locality = locality
        self.chunks = chunks
        self.periodic = periodic

    @eqx.filter_jit
    def moving_window(self, a):
        """Generate window to compute localized states."""
        size = int(self.in_dim / self.chunks + 2 * self.locality)
        starts = jnp.arange(len(a) - size + 1)[: self.chunks] * int(
            self.in_dim / self.chunks
        )
        return eqx.filter_vmap(
            lambda start: jax.lax.dynamic_slice(a, (start,), (size,))
        )(starts)

    @eqx.filter_jit
    def localize(self, in_state: Array) -> Array:
        """Generate parallel reservoir inputs from input state.

        Parameters
        ----------
        in_state : Array
            Input state, (shape=(in_dim,))

        Returns
        -------
        Array
            Parallel reservoir inputs, (shape=(chunks, chunk_size + 2*locality))
        """
        if len(in_state.shape) != 1:
            raise ValueError(
                "Only 1-dimensional localization is currently supported, detected a "
                f"{len(in_state.shape)}D field."
            )
        aug_state = jnp.hstack(
            [in_state[-self.locality :], in_state, in_state[: self.locality]]
        )
        if not self.periodic:
            aug_state = aug_state.at[: self.locality].set(aug_state[self.locality])
            aug_state = aug_state.at[-self.locality :].set(aug_state[-self.locality])
        return self.moving_window(aug_state)

    @eqx.filter_jit
    def embed(self, in_state: Array) -> Array:
        """Embed single state to reservoir dimensions.

        Parameters
        ----------
        in_state : Array
            Input state, (shape=(in_dim,)).

        Returns
        -------
        Array
            Embedded input to reservoir, (shape=(chunks, res_dim,)).
        """
        if in_state.shape != (self.in_dim,):
            raise ValueError("Incorrect dimension for input state.")
        localized_states = self.localize(in_state)

        return eqx.filter_vmap(jnp.matmul)(self.win, localized_states)

    def __call__(self, in_state: Array) -> Array:
        """Embed state to reservoir dimensions.

        Parameters
        ----------
        in_state : Array
            Input state, (shape=(in_dim,) or shape=(seq_len, in_dim)).

        Returns
        -------
        Array
            Embedded input to reservoir, (shape=(chunks, res_dim,) or
            shape=(seq_len, chunks, res_dim)).
        """
        if len(in_state.shape) == 1:
            to_ret = self.embed(in_state)
        elif len(in_state.shape) == 2:
            to_ret = self.batch_embed(in_state)
        else:
            raise ValueError(
                "Only 1-dimensional localization is currently supported, detected a "
                f"{len(in_state.shape) - 1}D field."
            )
        return to_ret
