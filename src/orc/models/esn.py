"""Classic ESN implementation with tanh nonlinearity and linear readout."""

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from orc.drivers import ESNDriver
from orc.embeddings import LinearEmbedding
from orc.rc import CRCForecasterBase, RCForecasterBase
from orc.readouts import LinearReadout, QuadraticReadout

jax.config.update("jax_enable_x64", True)


class ESNForecaster(RCForecasterBase):
    """
    Basic implementation of ESN for forecasting.

    Attributes
    ----------
    res_dim : int
        Reservoir dimension.
    data_dim : int
        Input/output dimension.
    driver : ESNDriver
        Driver implmenting the Echo State Network dynamics.
    readout : BaseReadout
        Trainable linear readout layer.
    embedding : LinearEmbedding
        Untrainable linear embedding layer.

    Methods
    -------
    force(in_seq, res_state)
        Teacher forces the reservoir with sequence in_seq and init. cond. res_state.
    forecast(fcast_len, res_state)
        Perform a forecast of fcast_len steps from res_state.
    set_readout(readout)
        Replace readout layer.
    set_embedding(embedding)
        Replace embedding layer.
    """

    res_dim: int
    data_dim: int

    def __init__(
        self,
        data_dim: int,
        res_dim: int,
        leak_rate: float = 0.6,
        bias: float = 1.6,
        embedding_scaling: float = 0.08,
        Wr_density: float = 0.02,
        Wr_spectral_radius: float = 0.8,
        dtype: type = jnp.float64,
        seed: int = 0,
        chunks: int = 1,
        locality: int = 0,
        quadratic: bool = False,
        periodic: bool = True,
        use_sparse_eigs: bool = True,
    ) -> None:
        """
        Initialize the ESN model.

        Parameters
        ----------
        data_dim : int
            Dimension of the input data.
        res_dim : int
            Dimension of the reservoir adjacency matrix Wr.
        leak_rate : float
            Integration leak rate of the reservoir dynamics.
        bias : float
            Bias term for the reservoir dynamics.
        embedding_scaling : float
            Scaling factor for the embedding layer.
        Wr_density : float
            Density of the reservoir adjacency matrix Wr.
        Wr_spectral_radius : float
            Largest eigenvalue of the reservoir adjacency matrix Wr.
        dtype : type
            Data type of the model (jnp.float64 is highly recommended).
        seed : int
            Random seed for generating the PRNG key for the reservoir computer.
        chunks : int
            Number of parallel reservoirs, must evenly divide data_dim.
        locality : int
            Overlap in adjacent parallel reservoirs.
        quadratic : bool
            Use quadratic nonlinearity in output, default False.
        periodic : bool
            Periodic BCs for embedding layer.
        use_sparse_eigs : bool
            Whether to use sparse eigensolver for setting the spectral radius of wr.
            Default is True, which is recommended to save memory and compute time. If
            False, will use dense eigensolver which may be more accurate.
        """
        # Initialize the random key and reservoir dimension
        self.res_dim = res_dim
        self.seed = seed
        self.data_dim = data_dim
        key = jax.random.PRNGKey(seed)
        key_driver, key_readout, key_embedding = jax.random.split(key, 3)

        # init in embedding, driver and readout
        embedding = LinearEmbedding(
            in_dim=data_dim,
            res_dim=res_dim,
            seed=key_embedding[0],
            scaling=embedding_scaling,
            chunks=chunks,
            locality=locality,
            periodic=periodic,
        )
        driver = ESNDriver(
            res_dim=res_dim,
            seed=key_driver[0],
            leak=leak_rate,
            bias=bias,
            density=Wr_density,
            spectral_radius=Wr_spectral_radius,
            chunks=chunks,
            dtype=dtype,
            use_sparse_eigs=use_sparse_eigs,
        )
        if quadratic:
            readout = QuadraticReadout(
                out_dim=data_dim, res_dim=res_dim, seed=key_readout[0], chunks=chunks
            )
        else:
            readout = LinearReadout(
                out_dim=data_dim, res_dim=res_dim, seed=key_readout[0], chunks=chunks
            )

        super().__init__(
            driver=driver,
            readout=readout,
            embedding=embedding,
            res_dim=res_dim,
            in_dim=data_dim,
            out_dim=data_dim,
            dtype=dtype,
            seed=seed,
        )
        self.chunks = chunks


class CESNForecaster(CRCForecasterBase):
    """
    Basic implementation of a Continuous ESN for forecasting.

    Attributes
    ----------
    res_dim : int
        Reservoir dimension.
    data_dim : int
        Input/output dimension.
    driver : ESNDriver
        Driver implementing the Echo State Network dynamics
        in continuous time.
    readout : BaseReadout
        Trainable linear readout layer.
    embedding : LinearEmbedding
        Untrainable linear embedding layer.

    Methods
    -------
    force(in_seq, res_state)
        Teacher forces the reservoir with sequence in_seq and init. cond. res_state.
    forecast(fcast_len, res_state)
        Perform a forecast of fcast_len steps from res_state.
    set_readout(readout)
        Replace readout layer.
    set_embedding(embedding)
        Replace embedding layer.
    """

    res_dim: int
    data_dim: int

    def __init__(
        self,
        data_dim: int,
        res_dim: int,
        time_const: float = 50.0,
        bias: float = 1.6,
        embedding_scaling: float = 0.08,
        Wr_density: float = 0.02,
        Wr_spectral_radius: float = 0.8,
        dtype: type = jnp.float64,
        seed: int = 0,
        chunks: int = 1,
        locality: int = 0,
        quadratic: bool = False,
        periodic: bool = True,
        use_sparse_eigs: bool = True,
        solver: diffrax.AbstractSolver = None,
        stepsize_controller: diffrax.AbstractAdaptiveStepSizeController = None,
    ) -> None:
        """
        Initialize the CESN model.

        Parameters
        ----------
        data_dim : int
            Dimension of the input data.
        res_dim : int
            Dimension of the reservoir adjacency matrix Wr.
        time_const : float
            Time constant of the reservoir dynamics.
        bias : float
            Bias term for the reservoir dynamics.
        embedding_scaling : float
            Scaling factor for the embedding layer.
        Wr_density : float
            Density of the reservoir adjacency matrix Wr.
        Wr_spectral_radius : float
            Largest eigenvalue of the reservoir adjacency matrix Wr.
        dtype : type
            Data type of the model (jnp.float64 is highly recommended).
        seed : int
            Random seed for generating the PRNG key for the reservoir computer.
        chunks : int
            Number of parallel reservoirs, must evenly divide data_dim.
        locality : int
            Overlap in adjacent parallel reservoirs.
        quadratic : bool
            Use quadratic nonlinearity in output, default False.
        periodic : bool
            Periodic BCs for embedding layer.
        use_sparse_eigs : bool
            Whether to use sparse eigensolver for setting the spectral radius of wr.
            Default is True, which is recommended to save memory and compute time. If
            False, will use dense eigensolver which may be more accurate.
        """
        # Initialize the random key and reservoir dimension
        self.res_dim = res_dim
        self.seed = seed
        self.data_dim = data_dim
        key = jax.random.PRNGKey(seed)
        key_driver, key_readout, key_embedding = jax.random.split(key, 3)

        # init in embedding, driver and readout
        embedding = LinearEmbedding(
            in_dim=data_dim,
            res_dim=res_dim,
            seed=key_embedding[0],
            scaling=embedding_scaling,
            chunks=chunks,
            locality=locality,
            periodic=periodic,
        )
        driver = ESNDriver(
            res_dim=res_dim,
            seed=key_driver[0],
            time_const=time_const,
            bias=bias,
            density=Wr_density,
            spectral_radius=Wr_spectral_radius,
            chunks=chunks,
            mode="continuous",
            dtype=dtype,
            use_sparse_eigs=use_sparse_eigs,
        )
        if quadratic:
            readout = QuadraticReadout(
                out_dim=data_dim, res_dim=res_dim, seed=key_readout[0], chunks=chunks
            )
        else:
            readout = LinearReadout(
                out_dim=data_dim, res_dim=res_dim, seed=key_readout[0], chunks=chunks
            )

        if solver is None:
            solver = diffrax.Tsit5()
        if stepsize_controller is None:
            stepsize_controller = diffrax.PIDController(rtol=1e-3,
                                                        atol=1e-6,
                                                        icoeff=1.0)

        super().__init__(
            driver=driver,
            readout=readout,
            embedding=embedding,
            res_dim=res_dim,
            in_dim=data_dim,
            out_dim=data_dim,
            dtype=dtype,
            seed=seed,
            solver=solver,
            stepsize_controller=stepsize_controller,
        )
        self.chunks = chunks


def _solve_single_ridge_reg(res_seq, target_seq, beta):
    """Solve a single matrix ridge regression problem."""
    lhs = res_seq.T @ res_seq + beta * jnp.eye(res_seq.shape[1], dtype=res_seq.dtype)
    rhs = res_seq.T @ target_seq
    cmat = jax.scipy.linalg.solve(lhs, rhs, assume_a="sym").T
    return cmat


# vmap ridge regression solver for parallel RC cases
_solve_all_ridge_reg = eqx.filter_vmap(_solve_single_ridge_reg, in_axes=eqx.if_array(1))


def train_ESNForecaster(
    model: ESNForecaster,
    train_seq: Array,
    target_seq: Array = None,
    spinup: int = 0,
    initial_res_state: Array = None,
    beta: float = 8e-8,
) -> tuple[ESNForecaster, Array]:
    """Training function for ESNForecaster.

    Parameters
    ----------
    model : ESNForecaster
        ESNForecaster model to train.
    train_seq : Array
        Training input sequence for reservoir, (shape=(seq_len, data_dim)).
    target_seq : Array
        Target sequence for training reservoir, (shape=(seq_len, data_dim)).
    initial_res_state : Array
        Initial reservoir state, (shape=(chunks, res_dim,)).
    spinup : int
        Initial transient of reservoir states to discard.
    beta : float
        Tikhonov regularization parameter.

    Returns
    -------
    model : ESNForecaster
        Trained ESN model.
    res_seq : Array
        Training sequence of reservoir states.
    """
    # Check that model is an ESN
    if not isinstance(model, ESNForecaster):
        raise TypeError("Model must be an ESNForecaster.")

    # check that spinup is less than the length of the training sequence
    if spinup >= train_seq.shape[0]:
        raise ValueError(
            "spinup must be less than the length of the training sequence.")

    if initial_res_state is None:
        initial_res_state = jnp.zeros(
            (
                model.embedding.chunks,
                model.res_dim,
            ),
            dtype=model.dtype,
        )

    if target_seq is None:
        target_seq = train_seq[1:, :]
        train_seq = train_seq[:-1, :]

    res_seq = model.force(train_seq, initial_res_state)
    if isinstance(model.readout, QuadraticReadout):
        res_seq_train = res_seq.at[:, :, ::2].set(res_seq[:, :, ::2] ** 2)
    else:
        res_seq_train = res_seq

    cmat = _solve_all_ridge_reg(
        res_seq_train[spinup:],
        target_seq[spinup:].reshape(res_seq[spinup:].shape[0], res_seq.shape[1], -1),
        beta,
    )

    def where(m):
        return m.readout.wout

    model = eqx.tree_at(where, model, cmat)

    return model, res_seq


def train_CESNForecaster(
    model: CESNForecaster,
    train_seq: Array,
    t_train: Array,
    target_seq: Array = None,
    spinup: int = 0,
    initial_res_state: Array = None,
    beta: float = 8e-8,
) -> tuple[CESNForecaster, Array]:
    """Training function for CESNForecaster.

    Parameters
    ----------
    model : CESNForecaster
        CESNForecaster model to train.
    train_seq : Array
        Training input sequence for reservoir, (shape=(seq_len, data_dim)).
    t_train : Array
        time vector corresponding to the training sequence, (shape=(seq_len,)).
    target_seq : Array
        Target sequence for training reservoir, (shape=(seq_len, data_dim)).
    initial_res_state : Array
        Initial reservoir state, (shape=(chunks, res_dim,)).
    spinup : int
        Initial transient of reservoir states to discard.
    beta : float
        Tikhonov regularization parameter.

    Returns
    -------
    model : CESNForecaster
        Trained CESN model.
    res_seq : Array
        Training sequence of reservoir states.
    """
    # check that model is continuous
    if not isinstance(model, CESNForecaster):
        raise TypeError("Model must be a CESNForecaster.")

    # check that train_seq and t_train have the same length
    if train_seq.shape[0] != t_train.shape[0]:
        raise ValueError("train_seq and t_train must have the same length.")

    # check that spinup is less than the length of the training sequence
    if spinup >= train_seq.shape[0]:
        raise ValueError(
            "spinup must be less than the length of the training sequence.")

    if initial_res_state is None:
        initial_res_state = jnp.zeros(
            (
                model.embedding.chunks,
                model.res_dim,
            ),
            dtype=model.dtype,
        )

    if target_seq is None:
        target_seq = train_seq[1:, :]
        train_seq = train_seq[:-1, :]
        t_train = t_train[:-1]

    res_seq = model.force(train_seq, initial_res_state, ts=t_train)
    if isinstance(model.readout, QuadraticReadout):
        res_seq_train = res_seq.at[:, :, ::2].set(res_seq[:, :, ::2] ** 2)
    else:
        res_seq_train = res_seq

    cmat = _solve_all_ridge_reg(
        res_seq_train[spinup:],
        target_seq[spinup:].reshape(res_seq[spinup:].shape[0], res_seq.shape[1], -1),
        beta,
    )

    def where(m):
        return m.readout.wout

    model = eqx.tree_at(where, model, cmat)

    return model, res_seq
