"""Classic ESN implementation with tanh nonlinearity and linear readout."""

import jax
import jax.numpy as jnp

from orc.drivers import ESNDriver
from orc.embeddings import LinearEmbedding
from orc.rc import ReservoirComputerBase
from orc.readouts import LinearReadout

jax.config.update("jax_enable_x64", True)

class ESN(ReservoirComputerBase):
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
    readout : LinearReadout
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


    def __init__(self,
                 data_dim: int,
                 res_dim: int,
                 leak_rate: float = 0.6,
                 bias: float = 1.6,
                 embedding_scaling: float = 0.08,
                 Wr_density: float = 0.02,
                 Wr_spectral_radius: float = 0.8,
                 dtype: type = jnp.float64,
                 seed: int = 0,
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
        """
        # Initialize the random key and reservoir dimension
        self.res_dim = res_dim
        self.seed = seed
        key = jax.random.PRNGKey(seed)
        key_driver, key_readout, key_embedding = jax.random.split(key, 3)

        # init in embedding, driver and readout
        embedding = LinearEmbedding(in_dim=data_dim, res_dim=res_dim, key=key_embedding, scaling=embedding_scaling)
        driver = ESNDriver(res_dim=res_dim, key=key_driver, leak=leak_rate, bias=bias, density=Wr_density, spec_rad=Wr_spectral_radius)
        readout = LinearReadout(out_dim=data_dim, res_dim=res_dim, key=key_readout)
        super().__init__(
            driver=driver,
            readout=readout,
            embedding=embedding,
            data_dim=data_dim,
            dtype=dtype,
            seed=seed,
        )
