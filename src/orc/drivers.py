"""Define base class for reservoir drivers and implement common architectures."""

import warnings
from abc import ABC, abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random
from equinox.nn._misc import default_init
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
        Reservoir dimensione
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
        """Advance the reservoir given projected inputs and current state.

        If driver supports parallel reservoirs, this method needs to be overwritten
        to accomodate shape handling. It can be overwritten with:

        ```
        def __call__(self, proj_vars, res_state):
            retur self._par_call(proj_vars, res_state)

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
        return self.advance(proj_vars, res_state)

    def _par_call(self, proj_vars: Array, res_state: Array) -> Array:
        """Advance the parallel reservoir given projected inputs and current state.

        Parameters
        ----------
        proj_vars : Array
            Projected inputs to reservoir.
        res_state : Array
            Initial reservoir state.

        Returns
        -------
        Array
            Updated reservoir state.
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

        # Compute eigenvalues - batch if requested to avoid memory issues
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
        self.wr = spectral_radius * (sp_mat / eigs[:, None, None])
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
                + self.sparse_ops(
                    self.wr, res_state, proj_vars, self.bias * jnp.ones_like(proj_vars)
                )
            )
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
        return self._par_call(proj_vars, res_state)


class _ParGRUCell(eqx.Module):
    """Parallel GRU cell implementation for reservoir computing.

    Attributes
    ----------
    input_size : int
        Input dimension size.
    hidden_size : int
        Hidden state dimension size.
    chunks : int
        Number of parallel GRU cells.
    weight_ih : Array
        Input-to-hidden weights, (shape=(chunks, 3 * hidden_size, input_size)).
    weight_hh : Array
        Hidden-to-hidden weights, (shape=(chunks, 3 * hidden_size, hidden_size)).
    bias : Array | None
        Input gate biases, (shape=(chunks, 3 * hidden_size)).
    bias_n : Array | None
        New gate biases, (shape=(chunks, hidden_size)).
    use_bias : bool
        Whether to use bias parameters.
    dtype : Float
        Dtype, default jnp.float64.

    Methods
    -------
    __call__(input, hidden)
        Apply GRU cell computation to inputs and hidden states.
    """

    input_size: int
    hidden_size: int
    chunks: int
    weight_ih: Array
    weight_hh: Array
    bias: Array | None
    bias_n: Array | None
    use_bias: bool = True
    dtype: Float = jnp.float64

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        chunks: int,
        use_bias: bool = True,
        *,
        seed: int,
        dtype: Float = jnp.float64,
    ):
        key = jax.random.key(seed)
        ihkey, hhkey, bkey, bkey2 = jax.random.split(key, 4)
        lim = jnp.sqrt(1 / hidden_size)

        ihshape = (chunks, 3 * hidden_size, input_size)
        self.weight_ih = default_init(ihkey, ihshape, dtype, lim)
        hhshape = (chunks, 3 * hidden_size, hidden_size)
        self.weight_hh = default_init(hhkey, hhshape, dtype, lim)
        if use_bias:
            self.bias = default_init(
                bkey,
                (
                    chunks,
                    3 * hidden_size,
                ),
                dtype,
                lim,
            )
            self.bias_n = default_init(
                bkey2,
                (
                    chunks,
                    hidden_size,
                ),
                dtype,
                lim,
            )
        else:
            self.bias = None
            self.bias_n = None

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.dtype = dtype
        self.chunks = chunks

    @eqx.filter_jit
    @eqx.filter_vmap
    def __call__(self, input_state: Array, hidden_state: Array):
        """Logic for advancing parallel GRU states."""
        if self.use_bias:
            bias = self.bias
            bias_n = self.bias_n
        else:
            bias = 0
            bias_n = 0
        igates = jnp.split(self.weight_ih @ input_state + bias, 3)
        hgates = jnp.split(self.weight_hh @ hidden_state, 3)
        reset = jax.nn.sigmoid(igates[0] + hgates[0])
        inp = jax.nn.sigmoid(igates[1] + hgates[1])
        new = jax.nn.tanh(igates[2] + reset * (hgates[2] + bias_n))
        return new + inp * (hidden_state - new)


class GRUDriver(DriverBase):
    """GRU-based reservoir driver.

    Attributes
    ----------
    res_dim : int
        Reservoir dimension.
    gru : _ParGRUCell
        Parallel GRU cell module.
    chunks : int
        Number of parallel reservoirs.
    dtype : Float
        Dtype, default jnp.float64.

    Methods
    -------
    advance(res_state, in_state)
        Updated reservoir state using GRU dynamics.
    __call__(res_state, in_state)
        Batched or single update to reservoir state.
    """

    gru: _ParGRUCell
    chunks: int

    def __init__(
        self,
        res_dim,
        chunks,
        seed=0,
        *,
        use_bias=True,
    ):
        super().__init__(res_dim=res_dim)
        self.gru = _ParGRUCell(res_dim, res_dim, chunks, use_bias=use_bias, seed=seed)
        self.chunks = chunks

    def advance(self, res_state, in_state):
        """Advance reservoir state.

        Parameters
        ----------
        proj_vars : Array
            Reservoir projected inputs, (shape=(chunks, res_dim).
        res_state : Array
            Current reservoir state, (shape=(chunks, res_dim).

        Returns
        -------
        Array
            Sequence of reservoir states, (shape=(chunks, res_dim,) or
            shape=(seq_len, chunks, res_dim)).
        """
        return self.gru(in_state, res_state)

    @eqx.filter_jit
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
        return self._par_call(proj_vars, res_state)
