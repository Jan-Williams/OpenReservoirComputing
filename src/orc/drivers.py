"""Define base class for reservoir drivers and implement common architectures."""

import warnings
from abc import ABC, abstractmethod

import equinox as eqx
import jax
import jax.experimental.sparse
import jax.numpy as jnp
import jax.random
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
            Projected inputs to reservoir, (shape=(res_dim,)).
        res_state : Array
            Initial reservoir state, (shape=(res_dim,)).

        Returns
        -------
        Array
            Updated reservoir state, (shape=(res_dim,)).
        """
        pass

    def batch_advance(self, proj_vars: Array, res_state: Array) -> Array:
        """
        Batch advance the reservoir given projected inputs and current state.

        Parameters
        ----------
        proj_vars : Array
            Reservoir projected inputs, (shape=(batch_size, res_dim,)).
        res_state : Array
            Reservoir state, (shape=(batch_size, res_dim,)).

        Returns
        -------
        Array
            Updated reservoir state, (shape=(batch_size, res_dim,)).
        """
        return eqx.filter_vmap(self.advance)(proj_vars, res_state)


class ESNDriver(DriverBase):
    """Standard implementation of ESN reservoir with tanh nonlinearity.

    Attributes
    ----------
    res_dim : int
        Reservoir dimension.
    wr : Array
        Reservoir update matrix, (shape=(res_dim, res_dim,)).
    leak : float
        Leak rate parameter.
    spec_rad : float
        Spectral radius of wr.
    density : float
        Density of wr.
    bias : float
        Additive bias in tanh nonlinearity.
    dtype : Float
        Dtype, default jnp.float64.

    Methods
    -------
    advance(proj_vars, res_state) -> updated reservoir state
    """

    res_dim: int
    leak: float
    spec_rad: float
    density: float
    bias: float
    dtype: Float
    wr: Array

    def __init__(
        self,
        res_dim: int,
        leak: float = 0.6,
        spec_rad: float = 0.8,
        density: float = 0.02,
        bias: float = 1.6,
        dtype: Float = jnp.float64,
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
        spec_rad : float
            Spectral radius of wr.
        density : float
            Density of wr.
        bias : float
            Additive bias in tanh nonlinearity.
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
        self.spec_rad = spec_rad
        self.density = density
        self.bias = bias
        self.dtype = dtype
        key = jax.random.key(seed)
        if spec_rad <= 0:
            raise ValueError("Spectral radius must be positve.")
        if leak < 0 or leak > 1:
            raise ValueError("Leak rate must satisfy 0 < leak < 1.")
        if density < 0 or density > 1:
            raise ValueError("Density must satisfy 0 < density < 1.")
        _, wr_key = jax.random.split(key)

        # check res_dim size for eigensolver choice
        if res_dim < 100 and use_sparse_eigs:
            use_sparse_eigs = False
            warnings.warn(
                "Reservoir dimension is less than 100, using dense " \
                "eigensolver for spectral radius.", stacklevel=2
            )

        # generate initial wr
        wr = jax.experimental.sparse.random_bcoo(key=wr_key,
                                                 shape=(res_dim, res_dim),
                                                 nse=density,
                                                 dtype=dtype,
                                                 generator = jax.random.normal)
        del wr_key # consumed by wr

        # set wr spectral radius
        if use_sparse_eigs:
            eig_max = jnp.abs(max_eig_arnoldi(wr))
        else:
            wr_dense = wr.todense()
            eig_max = jnp.max(jnp.abs(jnp.linalg.eigvals(wr_dense)))

        self.wr = wr * (spec_rad / eig_max)
        self.dtype = dtype

    def advance(self, proj_vars: Array, res_state: Array) -> Array:
        """Advance the reservoir state.

        Parameters
        ----------
        proj_vars : Array
            Reservoir projected inputs, (shape=(res_dim,)).
        res_state : Array
            Reservoir state, (shape=(res_dim,)).

        Returns
        -------
        res_next : Array
            Reservoir state, (shape=(res_dim,)).
        """
        if proj_vars.shape != (self.res_dim,) or res_state.shape != (self.res_dim,):
            raise ValueError("proj_vars and res_state must both have shape (res_dim,).")
        res_next = jnp.tanh(
            self.wr @ res_state + proj_vars + self.bias * jnp.ones(self.res_dim)
        )
        res_next = self.leak * res_next + (1 - self.leak) * res_state
        return res_next
