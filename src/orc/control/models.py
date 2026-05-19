"""Discrete ESN controller implementation."""

import jax
import jax.numpy as jnp

from orc.control.base import RCControllerBase
from orc.drivers import ESNDriver
from orc.embeddings import LinearEmbedding
from orc.readouts import LinearReadout, QuadraticReadout


class ESNController(RCControllerBase):
    """
    Basic implementation of ESN for control tasks.

    Attributes
    ----------
    res_dim : int
        Reservoir dimension.
    data_dim : int
        System input/output dimension.
    control_dim : int
        Control input dimension.
    driver : ESNDriver
        Driver implementing the Echo State Network dynamics.
    readout : ReadoutBase
        Trainable linear readout layer.
    embedding : LinearEmbedding
        Untrainable linear embedding layer for [input, control] concatenation.
    alpha_1 : float
        Weight for trajectory deviation penalty in control optimization.
    alpha_2 : float
        Weight for control magnitude penalty in control optimization.
    alpha_3 : float
        Weight for control derivative penalty in control optimization.

    Methods
    -------
    force(in_seq, control_seq, res_state)
        Teacher forces the reservoir with input and control sequences.
    apply_control(control_seq, fcast_len, res_state)
        Apply a predefined control sequence in closed-loop.
    set_readout(readout)
        Replace readout layer.
    set_embedding(embedding)
        Replace embedding layer.
    """

    res_dim: int
    data_dim: int
    control_dim: int

    def __init__(
        self,
        data_dim: int,
        control_dim: int,
        res_dim: int,
        leak_rate: float = 0.6,
        bias: float = 1.6,
        embedding_scaling: float = 0.08,
        Wr_density: float = 0.02,
        Wr_spectral_radius: float = 0.8,
        dtype: type = jnp.float64,
        alpha_1: float = 100,
        alpha_2: float = 1,
        alpha_3: float = 5,
        seed: int = 0,
        quadratic: bool = False,
        use_sparse_eigs: bool = True,
    ) -> None:
        """
        Initialize the ESN controller.

        Parameters
        ----------
        data_dim : int
            Dimension of the system input/output data.
        control_dim : int
            Dimension of the control input.
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
        alpha_1 : float
            Weight for trajectory deviation penalty in control optimization.
        alpha_2 : float
            Weight for control magnitude penalty in control optimization.
        alpha_3 : float
            Weight for control derivative penalty in control optimization.
        seed : int
            Random seed for generating the PRNG key for the reservoir computer.
        quadratic : bool
            Use quadratic nonlinearity in output, default False.
        use_sparse_eigs : bool
            Whether to use sparse eigensolver for setting the spectral radius of wr.
            Default is True, which is recommended to save memory and compute time. If
            False, will use dense eigensolver which may be more accurate.
        """
        # Initialize the random key and reservoir dimension
        self.res_dim = res_dim
        self.seed = seed
        self.data_dim = data_dim
        self.control_dim = control_dim
        key = jax.random.PRNGKey(seed)
        key_driver, key_readout, key_embedding = jax.random.split(key, 3)

        # Embedding accepts concatenated [input, control] vectors
        # Total input dimension is data_dim + control_dim
        embedding = LinearEmbedding(
            in_dim=data_dim + control_dim,
            res_dim=res_dim,
            seed=key_embedding[0],
            scaling=embedding_scaling,
        )
        driver = ESNDriver(
            res_dim=res_dim,
            seed=key_driver[0],
            leak=leak_rate,
            bias=bias,
            density=Wr_density,
            spectral_radius=Wr_spectral_radius,
            dtype=dtype,
            use_sparse_eigs=use_sparse_eigs,
        )
        if quadratic:
            readout = QuadraticReadout(
                out_dim=data_dim, res_dim=res_dim, seed=key_readout[0]
            )
        else:
            readout = LinearReadout(
                out_dim=data_dim, res_dim=res_dim, seed=key_readout[0]
            )

        super().__init__(
            driver=driver,
            readout=readout,
            embedding=embedding,
            in_dim=data_dim,
            control_dim=control_dim,
            dtype=dtype,
            seed=seed,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            alpha_3=alpha_3
        )
