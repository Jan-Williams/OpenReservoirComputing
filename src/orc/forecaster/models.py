"""Discrete and continuous ESN implementations with standard driver."""

import diffrax
import jax
import jax.numpy as jnp

from orc.drivers import ParallelESNDriver
from orc.embeddings import EnsembleLinearEmbedding, ParallelLinearEmbedding
from orc.forecaster.base import CRCForecasterBase, RCForecasterBase
from orc.readouts import (
    EnsembleLinearReadout,
    ParallelLinearReadout,
    ParallelQuadraticReadout,
)

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
    driver : ParallelESNDriver
        Driver implmenting the Echo State Network dynamics.
    readout : BaseReadout
        Trainable linear readout layer.
    embedding : ParallelLinearEmbedding
        Untrainable linear embedding layer.

    Methods
    -------
    force(in_seq, res_state)
        Teacher forces the reservoir with sequence in_seq and init. cond. res_state.
    forecast(fcast_len, res_state)
        Perform a forecast of fcast_len steps from res_state.
    forecast_from_IC(fcast_len, spinup_data)
        Forecast from a sequence of spinup data.
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
        embedding = ParallelLinearEmbedding(
            in_dim=data_dim,
            res_dim=res_dim,
            seed=key_embedding[0],
            scaling=embedding_scaling,
            chunks=chunks,
            locality=locality,
            periodic=periodic,
        )
        driver = ParallelESNDriver(
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
            readout = ParallelQuadraticReadout(
                out_dim=data_dim, res_dim=res_dim, seed=key_readout[0], chunks=chunks
            )
        else:
            readout = ParallelLinearReadout(
                out_dim=data_dim, res_dim=res_dim, seed=key_readout[0], chunks=chunks
            )

        super().__init__(
            driver=driver,
            readout=readout,
            embedding=embedding,
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
    driver : ParallelESNDriver
        Driver implementing the Echo State Network dynamics
        in continuous time.
    readout : BaseReadout
        Trainable linear readout layer.
    embedding : ParallelLinearEmbedding
        Untrainable linear embedding layer.

    Methods
    -------
    force(in_seq, res_state)
        Teacher forces the reservoir with sequence in_seq and init. cond. res_state.
    forecast(fcast_len, res_state)
        Perform a forecast of fcast_len steps from res_state.
    forecast_from_IC(fcast_len, spinup_data)
        Forecast from a sequence of spinup data.
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
        embedding = ParallelLinearEmbedding(
            in_dim=data_dim,
            res_dim=res_dim,
            seed=key_embedding[0],
            scaling=embedding_scaling,
            chunks=chunks,
            locality=locality,
            periodic=periodic,
        )
        driver = ParallelESNDriver(
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
            readout = ParallelQuadraticReadout(
                out_dim=data_dim, res_dim=res_dim, seed=key_readout[0], chunks=chunks
            )
        else:
            readout = ParallelLinearReadout(
                out_dim=data_dim, res_dim=res_dim, seed=key_readout[0], chunks=chunks
            )

        if solver is None:
            solver = diffrax.Tsit5()
        if stepsize_controller is None:
            stepsize_controller = diffrax.PIDController(
                rtol=1e-3, atol=1e-6, icoeff=1.0
            )

        super().__init__(
            driver=driver,
            readout=readout,
            embedding=embedding,
            dtype=dtype,
            seed=seed,
            solver=solver,
            stepsize_controller=stepsize_controller,
        )
        self.chunks = chunks


class EnsembleESNForecaster(RCForecasterBase):
    """
    Ensembled ESNs for forecasting.

    Attributes
    ----------
    res_dim : int
        Reservoir dimension.
    data_dim : int
        Input/output dimension.
    driver : ParallelESNDriver
        Driver implmenting the Echo State Network dynamics.
    readout : EnsembleLinearReadout
        Trainable linear readout layer.
    embedding : EnsembleLinearEmbedding
        Untrainable linear embedding layer.

    Methods
    -------
    force(in_seq, res_state)
        Teacher forces the reservoir with sequence in_seq and init. cond. res_state.
    forecast(fcast_len, res_state)
        Perform a forecast of fcast_len steps from res_state.
    forecast_from_IC(fcast_len, spinup_data)
        Forecast from a sequence of spinup data.
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
        embedding = EnsembleLinearEmbedding(
            in_dim=data_dim,
            res_dim=res_dim,
            seed=key_embedding[0],
            scaling=embedding_scaling,
            chunks=chunks,
        )
        driver = ParallelESNDriver(
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

        readout = EnsembleLinearReadout(
            out_dim=data_dim, res_dim=res_dim, seed=key_readout[0], chunks=chunks
        )

        super().__init__(
            driver=driver,
            readout=readout,
            embedding=embedding,
            dtype=dtype,
            seed=seed,
        )
        self.chunks = chunks
