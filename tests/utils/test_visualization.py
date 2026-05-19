import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from orc.utils import visualization as vis


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def sample_time_series():
    t = np.linspace(0, 10, 100)
    return np.column_stack((np.sin(t), np.cos(t)))


@pytest.fixture
def sample_spatiotemporal():
    t = np.linspace(0, 10, 50)
    x = np.linspace(0, 1, 30)
    T, X = np.meshgrid(t, x, indexing="ij")
    return np.sin(T) * np.cos(2 * np.pi * X)


def test_plot_time_series_basic(sample_time_series):
    vis.plot_time_series(sample_time_series)


def test_plot_time_series_with_options(sample_time_series):
    t = np.linspace(0, 10, 100)
    vis.plot_time_series(
        [sample_time_series, sample_time_series],
        t=t,
        time_series_labels=["Data 1", "Data 2"],
        line_formats=["-", "--"],
        state_var_names=["Var 1", "Var 2"],
        t_lim=8,
        title="Test Plot",
    )


def test_plot_time_series_with_jax():
    t = jnp.linspace(0, 10, 100)
    data = jnp.column_stack((jnp.sin(t), jnp.cos(t)))
    vis.plot_time_series(data)


def test_imshow_1D_spatiotemp_basic(sample_spatiotemporal):
    vis.imshow_1D_spatiotemp(sample_spatiotemporal, 10)


def test_imshow_1D_spatiotemp_with_options(sample_spatiotemporal):
    vis.imshow_1D_spatiotemp(
        sample_spatiotemporal,
        tN=10,
        domain=(0, 2),
        figsize=(15, 5),
        title="Test Spatiotemporal Plot",
        x_label="Time",
        cmap="viridis",
    )


def test_input_validation():
    with pytest.raises(TypeError):
        vis.plot_time_series("not an array")

    with pytest.raises(TypeError):
        vis.plot_time_series(np.array([1, 2, 3]))

    with pytest.raises(TypeError):
        vis.imshow_1D_spatiotemp("not an array", 10)

    with pytest.raises(TypeError):
        vis.imshow_1D_spatiotemp(np.array([1, 2, 3]), 10)
