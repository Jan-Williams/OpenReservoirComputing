"""Define base class for reservoir drivers and implement common architectures."""

import warnings
from abc import ABC, abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random
from jax.experimental import sparse
from jaxtyping import Array, Float

from orc.utils import max_eig_arnoldi

jax.config.update("jax_enable_x64", True)


class DriverBase(eqx.Module, ABC):
    """
    Base class dictating API for all implemented reservoir drivers.

    Attributes
    ----------
    res_dim : int
        Reservoir dimensionxe
    dtype : Float
        Dtype for model, jnp.float64 or jnp.float32.

    Methods
    -------
    advance(proj_vars, res_state)
        Advance reservoir according to proj_vars.
    batch_advance(proj_vars, res_state)
        Advance batch of reservoir states according to proj_vars.
    """

    res_dim: int
    dtype: Float

    def __init__(self, res_dim, dtype=jnp.float64):
        """Ensure reservoir dim and dtype are correct type."""
        self.res_dim = res_dim
        if not isinstance(res_dim, int):
            raise TypeError("Reservoir dimension res_dim must be an integer.")
        self.dtype = dtype
        if not (dtype == jnp.float64 or dtype == jnp.float32):
            raise TypeError("dtype must be jnp.float64 or jnp.float32.")

    @abstractmethod
    def advance(self, proj_vars: Array, res_state: Array) -> Array:
        """Advance the reservoir given projected inputs and current state.

        Parameters
        ----------
        proj_vars : Array
            Projected inputs to reservoir.
        res_state : Array
            Initial reservoir state.

        Returns
        -------
        Array
            Updated reservoir state, (shape=(res_dim,)).
        """
        pass

    @eqx.filter_jit
    def batch_advance(self, proj_vars: Array, res_state: Array) -> Array:
        """
        Batch advance the reservoir given projected inputs and current state.

        Parameters
        ----------
        proj_vars : Array
            Reservoir projected inputs.
        res_state : Array
            Reservoir state.

        Returns
        -------
        Array
            Updated reservoir state.
        """
        return eqx.filter_vmap(self.advance)(proj_vars, res_state)


class ESNDriver(DriverBase):
    """Standard implementation of ESN reservoir with tanh nonlinearity.

    Attributes
    ----------
    res_dim : int
        Reservoir dimension.
    wr : Array
        Reservoir update matrix, (shape=(chunks, res_dim, res_dim,)).
    leak : float
        Leak rate parameter.
    spectral_radius : float
        Spectral radius of wr.
    density : float
        Density of wr.
    bias : float
        Additive bias in tanh nonlinearity.
    chunks: int
        Number of parallel reservoirs.
    dtype : Float
        Dtype, default jnp.float64.

    Methods
    -------
    advance(proj_vars, res_state)
        Updated reservoir state.
    __call__(proj_vars, res_state)
        Batched or single update to reservoir state.
    """

    res_dim: int
    leak: float
    spectral_radius: float
    density: float
    bias: float
    dtype: Float
    wr: Array
    chunks: int
    mode: str
    gamma: float

    def __init__(
        self,
        res_dim: int,
        leak: float = 0.6,
        spectral_radius: float = 0.8,
        density: float = 0.02,
        bias: float = 1.6,
        dtype: Float = jnp.float64,
        chunks: int = 1,
        mode: str = "discrete",
        gamma: float = None,
        *,
        seed: int,
        use_sparse_eigs: bool = True
    ) -> None:
        """Initialize weight matrices.

        Parameters
        ----------
        res_dim : int
            Reservoir dimension.
        leak : float
            Leak rate parameter.
        spectral_radius : float
            Spectral radius of wr.
        density : float
            Density of wr.
        bias : float
            Additive bias in tanh nonlinearity.
        chunks: int
            Number of parallel reservoirs.
        dtype : Float
            Dtype, default jnp.float64.
        seed : int
            Random seed for generating the PRNG key for the reservoir computer.
        use_sparse_eigs : bool
            Whether to use sparse eigensolver for setting the spectral radius of wr.
            Default is True, which is recommended to save memory and compute time. If
            False, will use dense eigensolver which may be more accurate.
        """
        super().__init__(res_dim=res_dim, dtype=dtype)
        self.res_dim = res_dim
        self.leak = leak
        self.spectral_radius = spectral_radius
        self.density = density
        self.bias = bias
        self.dtype = dtype
        self.mode = mode 
        self.gamma = gamma 
        key = jax.random.key(seed)
        if spectral_radius <= 0:
            raise ValueError("Spectral radius must be positve.")
        if leak < 0 or leak > 1:
            raise ValueError("Leak rate must satisfy 0 < leak < 1.")
        if density < 0 or density > 1:
            raise ValueError("Density must satisfy 0 < density < 1.")
        subkey, wr_key = jax.random.split(key)

        # check res_dim size for eigensolver choice
        if res_dim < 100 and use_sparse_eigs:
            use_sparse_eigs = False
            warnings.warn(
                "Reservoir dimension is less than 100, using dense " \
                "eigensolver for spectral radius.", stacklevel=2
            )

        # generate all wr matricies
        sp_mat = sparse.random_bcoo(key=wr_key,
                                    shape=(chunks, res_dim, res_dim),
                                    n_batch=1,
                                    nse=density,
                                    dtype=dtype,
                                    generator=jax.random.normal)
        if use_sparse_eigs:
            eigs = jnp.abs(jax.vmap(max_eig_arnoldi)(sp_mat))
        else:
            dense_mat = sparse.bcoo_todense(sp_mat)
            eigs = jnp.max(jnp.abs(jnp.linalg.eigvals(dense_mat)), axis=1)
        self.wr = spectral_radius*(sp_mat / eigs[:, None, None])
        self.chunks = chunks
        self.dtype = dtype

    @eqx.filter_jit
    def advance(self, proj_vars: Array, res_state: Array) -> Array:
        """Advance the reservoir state.

        Parameters
        ----------
        proj_vars : Array
            Reservoir projected inputs, (shape=(chunks, res_dim,)).
        res_state : Array
            Reservoir state, (shape=(chunks, res_dim,)).

        Returns
        -------
        res_next : Array
            Reservoir state, (shape=(chunks, res_dim,)).
        """
        if proj_vars.shape != (self.chunks, self.res_dim):
            raise ValueError(f"Incorrect proj_var dimension, got {proj_vars.shape}")
        if self.mode == "continuous":
            return self.gamma * (-res_state + self.sparse_ops(self.wr, res_state, proj_vars, self.bias*jnp.ones_like(proj_vars)))
        else:
            return (
                self.leak
                * self.sparse_ops(
                    self.wr, res_state, proj_vars, self.bias * jnp.ones_like(proj_vars)
                )
                + (1 - self.leak) * res_state
            )

    @staticmethod
    @sparse.sparsify
    @jax.vmap
    def sparse_ops(wr, res_state, proj_vars, bias):
        """Dense operation to sparsify for advancing reservoir."""
        return jnp.tanh(wr @ res_state + proj_vars + bias)

    def __call__(self, proj_vars: Array, res_state: Array) -> Array:
        """Advance reservoir state.

        Parameters
        ----------
        proj_vars : Array
            Reservoir projected inputs, (shape=(chunks, res_dim) or
            shape=(seq_len, chunks, res_dim)).
        res_state : Array
            Current reservoir state, (shape=(chunks, res_dim) or
            shape=(seq_len, chunks, res_dim)).

        Returns
        -------
        Array
            Sequence of reservoir states, (shape=(chunks, res_dim,) or
            shape=(seq_len, chunks, res_dim)).
        """
        if len(proj_vars.shape) == 2:
            to_ret = self.advance(proj_vars, res_state)
        elif len(proj_vars.shape) == 3:
            to_ret = self.batch_advance(proj_vars, res_state)
        else:
            raise ValueError(
                "Only 1-dimensional localization is currently supported, detected a "
                f"{len(proj_vars.shape)}D field."
            )
        return to_ret
