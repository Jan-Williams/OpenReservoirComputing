from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from orc.utils import visualization as vis


@pytest.fixture
def sample_time_series():
    # Create a simple time series with 2 state variables
    t = np.linspace(0, 10, 100)
    u1 = np.sin(t)
    u2 = np.cos(t)
    return np.column_stack((u1, u2))

@pytest.fixture
def sample_spatiotemporal():
    # Create a simple spatiotemporal dataset (Nt=50, Nx=30)
    t = np.linspace(0, 10, 50)
    x = np.linspace(0, 1, 30)
    T, X = np.meshgrid(t, x, indexing='ij')
    U = np.sin(T) * np.cos(2 * np.pi * X)
    return U

@pytest.fixture
def sample_3d_series():
    # Create a simple 3D time series: shape (Nt=100)
    t = np.linspace(0, 10, 100)
    x = np.sin(t)
    y = np.cos(t)
    z = np.sin(t) * np.cos(0.5 * t)
    return np.column_stack((x, y, z))

@patch('matplotlib.pyplot.show')
def test_plot_time_series_basic(mock_show, sample_time_series):
    # Test with basic parameters
    vis.plot_time_series(sample_time_series)
    mock_show.assert_called_once()

@patch('matplotlib.pyplot.show')
def test_plot_time_series_with_options(mock_show, sample_time_series):
    # Test with optional parameters
    t = np.linspace(0, 10, 100)
    vis.plot_time_series(
        [sample_time_series, sample_time_series],
        t=t,
        time_series_labels=["Data 1", "Data 2"],
        line_formats=['-', '--'],
        state_var_names=["Var 1", "Var 2"],
        t_lim=8,
        title="Test Plot"
    )
    mock_show.assert_called_once()

@patch('matplotlib.pyplot.show')
def test_plot_time_series_with_jax(mock_show):
    # Test with JAX arrays
    t = jnp.linspace(0, 10, 100)
    data = jnp.column_stack((jnp.sin(t), jnp.cos(t)))
    vis.plot_time_series(data)
    mock_show.assert_called_once()

@patch('matplotlib.pyplot.show')
def test_imshow_1D_spatiotemp_basic(mock_show, sample_spatiotemporal):
    # Test basic functionality
    vis.imshow_1D_spatiotemp(sample_spatiotemporal, 10)
    mock_show.assert_called_once()

@patch('matplotlib.pyplot.show')
def test_imshow_1D_spatiotemp_with_options(mock_show, sample_spatiotemporal):
    # Test with various options
    vis.imshow_1D_spatiotemp(
        sample_spatiotemporal,
        tN=10,
        domain=(0, 2),
        figsize=(15, 5),
        title="Test Spatiotemporal Plot",
        x_label="Time",
        cmap="viridis"
    )
    mock_show.assert_called_once()

@patch('matplotlib.pyplot.show')
def test_plot_in_3d_state_space_basic(mock_show, sample_3d_series):
    # Test with basic parameters
    vis.plot_in_3D_state_space(sample_3d_series)
    mock_show.assert_called_once()

@patch('matplotlib.pyplot.show')
def test_plot_in_3d_state_space_with_options(mock_show, sample_3d_series):
    # Test with optional parameters
    vis.plot_in_3D_state_space(
        [sample_3d_series, sample_3d_series],
        time_series_labels=["Data 1", "Data 2"],
        line_formats=['-', '--'],
        state_var_names=["x1", "x2", "x3"],
        title="3D Attractor",
        linewidth=1.5,
    )
    mock_show.assert_called_once()

@patch('matplotlib.pyplot.show')
def test_plot_in_3d_state_space_with_jax(mock_show):
    # Test with JAX arrays
    t = jnp.linspace(0, 10, 100)
    x = jnp.sin(t)
    y = jnp.cos(t)
    z = jnp.sin(t) * jnp.cos(0.5 * t)
    data = jnp.column_stack((x, y, z))
    vis.plot_in_3D_state_space(data)
    mock_show.assert_called_once()

def test_input_validation():
    # Test input validation for all three functions
    with pytest.raises(TypeError):
        vis.plot_time_series("not an array")

    with pytest.raises(TypeError):
        vis.plot_time_series(np.array([1, 2, 3]))  # 1D array

    with pytest.raises(ValueError):
        a = np.zeros((10, 3))
        b = np.zeros((9, 3))
        vis.plot_time_series([a, b])  # mismatched shapes

    with pytest.raises(TypeError):
        vis.imshow_1D_spatiotemp("not an array", 10)

    with pytest.raises(TypeError):
        vis.imshow_1D_spatiotemp(np.array([1, 2, 3]), 10)  # 1D array

    with pytest.raises(TypeError):
        vis.plot_in_3D_state_space("not an array")

    with pytest.raises(TypeError):
        vis.plot_in_3D_state_space(np.array([1, 2, 3]))  # 1D array

    with pytest.raises(ValueError):
        a = np.zeros((10, 3))
        b = np.zeros((9, 3))
        vis.plot_in_3D_state_space([a, b])  # mismatched shapes
