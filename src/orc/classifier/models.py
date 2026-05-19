"""Discrete ESN classifier implementation."""

import jax
import jax.numpy as jnp

from orc.classifier.base import RCClassifierBase
from orc.drivers import ESNDriver
from orc.embeddings import LinearEmbedding
from orc.readouts import LinearReadout, QuadraticReadout



class ESNClassifier(RCClassifierBase):
    """
    Basic implementation of ESN for classification tasks.

    Attributes
    ----------
    res_dim : int
        Reservoir dimension.
    data_dim : int
        Input data dimension.
    n_classes : int
        Number of classification classes.
    driver : ESNDriver
        Driver implementing the Echo State Network dynamics.
    readout : ReadoutBase
        Trainable linear readout layer with out_dim=n_classes.
    embedding : LinearEmbedding
        Untrainable linear embedding layer.

    Methods
    -------
    force(in_seq, res_state)
        Teacher forces the reservoir with the input sequence.
    classify(in_seq, res_state)
        Classify an input sequence, returning class probabilities.
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
        n_classes: int,
        res_dim: int,
        leak_rate: float = 0.6,
        bias: float = 1.6,
        embedding_scaling: float = 0.08,
        Wr_density: float = 0.02,
        Wr_spectral_radius: float = 0.8,
        dtype: type = jnp.float64,
        seed: int = 0,
        quadratic: bool = False,
        use_sparse_eigs: bool = True,
        state_repr: str = "final",
    ) -> None:
        """
        Initialize the ESN classifier.

        Parameters
        ----------
        data_dim : int
            Dimension of the input data.
        n_classes : int
            Number of classification classes.
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
        quadratic : bool
            Use quadratic nonlinearity in output, default False.
        use_sparse_eigs : bool
            Whether to use sparse eigensolver for setting the spectral radius of wr.
            Default is True, which is recommended to save memory and compute time. If
            False, will use dense eigensolver which may be more accurate.
        state_repr : str
            Reservoir state representation for classification.
            "final" uses the last reservoir state, "mean" averages states after spinup.
        """
        # Initialize the random key and reservoir dimension
        self.res_dim = res_dim
        self.seed = seed
        self.data_dim = data_dim
        key = jax.random.PRNGKey(seed)
        key_driver, key_readout, key_embedding = jax.random.split(key, 3)

        embedding = LinearEmbedding(
            in_dim=data_dim,
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
                out_dim=n_classes, res_dim=res_dim, seed=key_readout[0]
            )
        else:
            readout = LinearReadout(
                out_dim=n_classes, res_dim=res_dim, seed=key_readout[0]
            )

        super().__init__(
            driver=driver,
            readout=readout,
            embedding=embedding,
            n_classes=n_classes,
            state_repr=state_repr,
            dtype=dtype,
            seed=seed,
        )
