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
        if self.chunks > 0:
            if len(proj_vars.shape) == 2:
                to_ret = self.advance(proj_vars, res_state)
            elif len(proj_vars.shape) == 3:
                to_ret = self.batch_advance(proj_vars, res_state)
        else:
            if len(proj_vars.shape) == 1:
                to_ret = self.advance(proj_vars, res_state)
            elif len(proj_vars.shape) == 2:
                to_ret = self.batch_advance(proj_vars, res_state)
        return to_ret


class ParallelESNDriver(DriverBase):
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
    mode : str
        Mode of reservoir update, either "discrete" or "continuous".
    time_const : float
        Time constant for continuous mode.
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
    time_const: float

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
        time_const: float = 50.0,
        *,
        seed: int,
        use_sparse_eigs: bool = True,
        eigenval_batch_size: int = None,
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
        mode : str
            Mode of reservoir update, either "discrete" or "continuous".
        time_const : float
            Time constant for continuous mode. Ignored in discrete mode.
        dtype : Float
            Dtype, default jnp.float64.
        seed : int
            Random seed for generating the PRNG key for the reservoir computer.
        use_sparse_eigs : bool
            Whether to use sparse eigensolver for setting the spectral radius of wr.
            Default is True, which is recommended to save memory and compute time. If
            False, will use dense eigensolver which may be more accurate.
        eigenval_batch_size : int
            Size of batches when batch_eigenvals. Default is None, which means no
            batch eigenvalue computation.
        """
        super().__init__(res_dim=res_dim, dtype=dtype)
        self.res_dim = res_dim
        self.leak = leak
        self.spectral_radius = spectral_radius
        self.density = density
        self.bias = bias
        self.dtype = dtype
        self.mode = mode
        self.time_const = time_const
        key = jax.random.key(seed)
        if spectral_radius <= 0:
            raise ValueError("Spectral radius must be positive.")
        if leak < 0 or leak > 1:
            raise ValueError("Leak rate must satisfy 0 < leak < 1.")
        if density < 0 or density > 1:
            raise ValueError("Density must satisfy 0 < density < 1.")
        if mode not in ["discrete", "continuous"]:
            raise ValueError("Mode must be either 'discrete' or 'continuous'.")
        if time_const <= 0:
            raise ValueError("Time constant must be positive.")
        subkey, wr_key = jax.random.split(key)

        # check res_dim size for eigensolver choice
        if res_dim < 100 and use_sparse_eigs:
            use_sparse_eigs = False
            warnings.warn(
                "Reservoir dimension is less than 100, using dense "
                "eigensolver for spectral radius.",
                stacklevel=2,
            )

        # generate all wr matricies
        sp_mat = sparse.random_bcoo(
            key=wr_key,
            shape=(chunks, res_dim, res_dim),
            n_batch=1,
            nse=density,
            dtype=dtype,
            generator=jax.random.normal,
        )

        self.wr = _spec_rad_normalization(
            sp_mat,
            spectral_radius=spectral_radius,
            eigenval_batch_size=eigenval_batch_size,
            use_sparse_eigs=use_sparse_eigs,
            chunks=chunks,
        )
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
            return self.time_const * (
                -res_state
                + _sparse_ops(
                    self.wr, res_state, proj_vars, self.bias * jnp.ones_like(proj_vars)
                )
            )
        else:
            return (
                self.leak
                * _sparse_ops(
                    self.wr, res_state, proj_vars, self.bias * jnp.ones_like(proj_vars)
                )
                + (1 - self.leak) * res_state
            )


class ESNDriver(ParallelESNDriver):
    """Standard implementation of single ESN reservoir with tanh nonlinearity.

    Attributes
    ----------
    res_dim : int
        Reservoir dimension.
    wr : Array
        Reservoir update matrix, (shape=(1, res_dim, res_dim,)).
    leak : float
        Leak rate parameter.
    spectral_radius : float
        Spectral radius of wr.
    density : float
        Density of wr.
    bias : float
        Additive bias in tanh nonlinearity.
    mode : str
        Mode of reservoir update, either "discrete" or "continuous".
    time_const : float
        Time constant for continuous mode.
    dtype : Float
        Dtype, default jnp.float64.

    Methods
    -------
    advance(proj_vars, res_state)
        Updated reservoir state.
    __call__(proj_vars, res_state)
        Batched or single update to reservoir state.
    """

    def __init__(
        self,
        res_dim: int,
        leak: float = 0.6,
        spectral_radius: float = 0.8,
        density: float = 0.02,
        bias: float = 1.6,
        dtype: Float = jnp.float64,
        mode: str = "discrete",
        time_const: float = 50.0,
        *,
        seed: int,
        use_sparse_eigs: bool = True,
        eigenval_batch_size: int = None,
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
        mode : str
            Mode of reservoir update, either "discrete" or "continuous".
        time_const : float
            Time constant for continuous mode. Ignored in discrete mode.
        dtype : Float
            Dtype, default jnp.float64.
        seed : int
            Random seed for generating the PRNG key for the reservoir computer.
        use_sparse_eigs : bool
            Whether to use sparse eigensolver for setting the spectral radius of wr.
            Default is True, which is recommended to save memory and compute time. If
            False, will use dense eigensolver which may be more accurate.
        eigenval_batch_size : int
            Size of batches when batch_eigenvals. Default is None, which means no
            batch eigenvalue computation.
        """
        super().__init__(
            res_dim=res_dim,
            leak=leak,
            spectral_radius=spectral_radius,
            density=density,
            bias=bias,
            dtype=dtype,
            mode=mode,
            time_const=time_const,
            seed=seed,
            use_sparse_eigs=use_sparse_eigs,
            eigenval_batch_size=eigenval_batch_size,
            chunks=1,
        )

    @eqx.filter_jit
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
        res_next = super().advance(proj_vars.reshape(1, -1), res_state.reshape(1, -1))
        res_next = jnp.squeeze(res_next)
        return res_next

    def __call__(self, proj_vars: Array, res_state: Array) -> Array:
        """Advance reservoir state.

        Parameters
        ----------
        proj_vars : Array
            Reservoir projected inputs, (shape=(res_dim,) or
            shape=(seq_len, res_dim)).
        res_state : Array
            Current reservoir state, (shape=(res_dim,) or
            shape=(seq_len, res_dim)).

        Returns
        -------
        Array
            Sequence of reservoir states, (shape=(res_dim,) or
            shape=(seq_len, res_dim)).
        """
        res_next = super().__call__(proj_vars[..., None, :], res_state[..., None, :])
        res_next = jnp.squeeze(res_next)
        return res_next


class ParallelTaylorDriver(DriverBase):
    """ESN driver with tanh nonlinearity, Taylor expanded.

    This class defines a driver according to the Taylor series expansion of
    ParallelESNDriver including the first ``n_terms`` terms with the leak rate
    leak=0. Only discrete time dynamics are supported.

    Attributes
    ----------
    n_terms : int
        Number of terms in Taylor series to include.
    res_dim : int
        Reservoir dimension.
    wr : Array
        Reservoir update matrix, (shape=(chunks, res_dim, res_dim,)).
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
    advance_full(proj_vars, res_state, terms)
        Updated reservoir state advanced according to full tanh nonlinearity.
    __call__(proj_vars, res_state)
        Batched or single update to reservoir state.
    """

    n_terms: int
    res_dim: int
    spectral_radius: float
    density: float
    bias: float
    dtype: Float
    wr: Array
    chunks: int

    def __init__(
        self,
        n_terms: int,
        res_dim: int,
        spectral_radius: float = 0.8,
        density: float = 0.02,
        bias: float = 1.6,
        dtype: Float = jnp.float64,
        chunks: int = 1,
        *,
        seed: int,
        use_sparse_eigs: bool = True,
        eigenval_batch_size: int = None,
    ) -> None:
        """Initialize weight matrices.

        Parameters
        ----------
        n_terms : int
            Number of terms to use in Taylor expansion.
        res_dim : int
            Reservoir dimension.
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
        eigenval_batch_size : int
            Size of batches when batch_eigenvals. Default is None, which means no
            batch eigenvalue computation.
        """
        super().__init__(res_dim=res_dim, dtype=dtype)
        self.n_terms = n_terms
        self.res_dim = res_dim
        self.spectral_radius = spectral_radius
        self.density = density
        self.bias = bias
        self.dtype = dtype
        key = jax.random.key(seed)
        if spectral_radius <= 0:
            raise ValueError("Spectral radius must be positive.")
        if density < 0 or density > 1:
            raise ValueError("Density must satisfy 0 < density < 1.")
        if n_terms > 5:
            raise ValueError("Taylor expansion is only supported up to 5th order.")
        subkey, wr_key = jax.random.split(key)

        # check res_dim size for eigensolver choice
        if res_dim < 100 and use_sparse_eigs:
            use_sparse_eigs = False
            warnings.warn(
                "Reservoir dimension is less than 100, using dense "
                "eigensolver for spectral radius.",
                stacklevel=2,
            )

        # generate all wr matricies
        sp_mat = sparse.random_bcoo(
            key=wr_key,
            shape=(chunks, res_dim, res_dim),
            n_batch=1,
            nse=density,
            dtype=dtype,
            generator=jax.random.normal,
        )

        self.wr = _spec_rad_normalization(
            sp_mat,
            spectral_radius=spectral_radius,
            eigenval_batch_size=eigenval_batch_size,
            use_sparse_eigs=use_sparse_eigs,
            chunks=chunks,
        )
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

        t = jnp.tanh(self.bias)
        s = 1 - t**2
        deltaz = sparse.sparsify(jax.vmap(jnp.matmul))(self.wr, res_state) + proj_vars
        const = t * jnp.ones((self.chunks, self.res_dim), dtype=self.dtype)
        linear_term = s * deltaz
        quadratic_term = (t**3 - t) * deltaz**2
        cubic_term = (-(t**4) + (4 / 3) * t**2 - (1 / 3)) * deltaz**3
        quartic_term = (t / 3) * (3 * t**4 - 5 * t**2 + 2) * deltaz**4
        quintic_term = (-(t**6) + 2 * t**4 - (17 / 15) * t**2 + (2 / 15)) * deltaz**5
        stacked = jnp.stack(
            [
                const,
                linear_term,
                quadratic_term,
                cubic_term,
                quartic_term,
                quintic_term,
            ],
            axis=0,
        )
        return jnp.sum(stacked[: self.n_terms + 1], axis=0)

    @eqx.filter_jit
    def advance_full(self, proj_vars: Array, res_state: Array) -> Array:
        """Advance the reservoir state according to full tanh dynamics.

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

        return _sparse_ops(
            self.wr, res_state, proj_vars, self.bias * jnp.ones_like(proj_vars)
        )


class TaylorDriver(ParallelTaylorDriver):
    """ESN driver with tanh nonlinearity, Taylor expanded.

    This class defines a driver according to the Taylor series expansion of
    ParallelESNDriver including the first ``n_terms`` terms with the leak rate
    leak=0. Only discrete time dynamics are supported.

    Attributes
    ----------
    n_terms : int
        Number of terms in Taylor series to include.
    res_dim : int
        Reservoir dimension.
    wr : Array
        Reservoir update matrix, (shape=(chunks, res_dim, res_dim,)).
    spectral_radius : float
        Spectral radius of wr.
    density : float
        Density of wr.
    bias : float
        Additive bias in tanh nonlinearity.
    dtype : Float
        Dtype, default jnp.float64.

    Methods
    -------
    advance(proj_vars, res_state)
        Updated reservoir state.
    __call__(proj_vars, res_state)
        Batched or single update to reservoir state.
    """

    def __init__(
        self,
        n_terms: int,
        res_dim: int,
        spectral_radius: float = 0.8,
        density: float = 0.02,
        bias: float = 1.6,
        dtype: Float = jnp.float64,
        *,
        seed: int,
        use_sparse_eigs: bool = True,
        eigenval_batch_size: int = None,
    ) -> None:
        """Initialize weight matrices.

        Parameters
        ----------
        n_terms : int
            Number of terms to use in Taylor expansion.
        res_dim : int
            Reservoir dimension.
        spectral_radius : float
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
        eigenval_batch_size : int
            Size of batches when batch_eigenvals. Default is None, which means no
            batch eigenvalue computation.
        """
        super().__init__(
            n_terms=n_terms,
            res_dim=res_dim,
            spectral_radius=spectral_radius,
            density=density,
            bias=bias,
            dtype=dtype,
            seed=seed,
            use_sparse_eigs=use_sparse_eigs,
            eigenval_batch_size=eigenval_batch_size,
            chunks=1,
        )

    @eqx.filter_jit
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
        res_next = super().advance(proj_vars.reshape(1, -1), res_state.reshape(1, -1))
        res_next = jnp.squeeze(res_next)
        return res_next

    def __call__(self, proj_vars: Array, res_state: Array) -> Array:
        """Advance reservoir state.

        Parameters
        ----------
        proj_vars : Array
            Reservoir projected inputs, (shape=(res_dim,) or
            shape=(seq_len, res_dim)).
        res_state : Array
            Current reservoir state, (shape=(res_dim,) or
            shape=(seq_len, res_dim)).

        Returns
        -------
        Array
            Sequence of reservoir states, (shape=(res_dim,) or
            shape=(seq_len, res_dim)).
        """
        res_next = super().__call__(proj_vars[..., None, :], res_state[..., None, :])
        res_next = jnp.squeeze(res_next)
        return res_next


@sparse.sparsify
@jax.vmap
def _sparse_ops(wr: Array, res_state: Array, proj_vars: Array, bias: Array):
    """Dense operation to sparsify for advancing reservoir."""
    return jnp.tanh(wr @ res_state + proj_vars + bias)


def _spec_rad_normalization(
    sp_mat: Array,
    spectral_radius: float,
    eigenval_batch_size: int | None = None,
    use_sparse_eigs: bool = True,
    chunks: int = 1,
):
    """Spectral radius normalization for jax sparse.bcoo matrices with n_batch=1."""
    if eigenval_batch_size is not None:
        batch_size = min(eigenval_batch_size, chunks)
        eigs_list = []

        for i in range(0, chunks, batch_size):
            end_idx = min(i + batch_size, chunks)
            batch_sp_mat = sp_mat[i:end_idx]

            if use_sparse_eigs:
                batch_eigs = jnp.abs(jax.vmap(max_eig_arnoldi)(batch_sp_mat))
            else:
                batch_dense_mat = sparse.bcoo_todense(batch_sp_mat)
                batch_eigs = jnp.max(
                    jnp.abs(jnp.linalg.eigvals(batch_dense_mat)), axis=1
                )

            eigs_list.append(batch_eigs)

        eigs = jnp.concatenate(eigs_list, axis=0)
    else:
        if use_sparse_eigs:
            eigs = jnp.abs(jax.vmap(max_eig_arnoldi)(sp_mat))
        else:
            dense_mat = sparse.bcoo_todense(sp_mat)
            eigs = jnp.max(jnp.abs(jnp.linalg.eigvals(dense_mat)), axis=1)
    sp_mat = spectral_radius * (sp_mat / eigs[:, None, None])
    return sp_mat


class GRUDriver(DriverBase):
    """Gated Recurrent Unit (GRU) based reservoir driver.

    This driver uses an Equinox GRUCell as the reservoir dynamics.

    Attributes
    ----------
    res_dim : int
        Reservoir dimension.
    gru : eqx.Module
        Equinox GRUCell module for reservoir updates.
    dtype : Float
        Dtype for model, jnp.float64 or jnp.float32.

    Methods
    -------
    advance(res_state, in_state)
        Advance reservoir state using GRU dynamics.
    """

    gru: eqx.Module
    chunks: int = 0

    def __init__(self, res_dim, *, seed):
        """Initialize GRU-based reservoir driver.

        Parameters
        ----------
        res_dim : int
            Reservoir dimension.
        seed : int
            Random seed for initializing GRU weights. Default is 0.
        """
        super().__init__(res_dim=res_dim)
        key = jax.random.key(seed)
        self.gru = eqx.nn.GRUCell(res_dim, res_dim, key=key)

    def advance(self, res_state, in_state):
        """Advance the reservoir state using GRU dynamics.

        Parameters
        ----------
        res_state : Array
            Current reservoir state, (shape=(res_dim,)).
        in_state : Array
            Projected inputs to reservoir, (shape=(res_dim,)).

        Returns
        -------
        Array
            Updated reservoir state, (shape=(res_dim,)).
        """
        return self.gru(in_state, res_state)
