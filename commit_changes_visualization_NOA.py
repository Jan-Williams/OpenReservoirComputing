"""Visualization utilities for plotting time series and spatiotemporal data."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def plot_time_series(U_lst,
                     t=None,
                     time_series_labels=None,
                     line_formats = None,
                     state_var_names=None,
                     t_lim = None,
                     figsize = (20,8),
                     x_label = r'$t$',
                     title = None,
                     **plot_kwargs):
    """Plot time series data with separate panels for each state variable.

    Parameters
    ----------
    U_lst : 2D array or list of 2D arrays
        If a 2D array, shape should be (Nt, Nu) where Nu is the number of state
        variables and Nt is the number of time points.If a list of 2D arrays,
        each array should have shape (Nt, Nu) and represent different time series.
    t : 1D array, optional
        1D array of time points. If None, the time points will be assumed to be
        evenly spaced from 0 to Nt-1.
    time_series_labels : list of strings, optional
        List of strings containing the labels for each time series to be shown in
        a legend. If None, no labels will be shown.
    line_formats : list of strings, optional
        List of strings containing the line formats for each time series. If None,
        default line format will be used.
    state_var_names : list of strings, optional
        List of strings containing the names of the state variables. If None,
        no y-axis labels will be shown.
    t_lim : tuple, optional
        Limit for the x-axis. If None, the x-axis will be set to the full
        range of time points.
    figsize : tuple, optional
        Size of the figure to be created. Default is (20, 8).
    x_label : string, optional
        Label for the x-axis. Default is r'$t$'.
    title : string, optional
        Title of the plot. If None, no title is shown.
    plot_kwargs : dict, optional
        Additional arguments to pass to the plot function.
    """
    # Input validation
    if not isinstance(U_lst, list):
        if not isinstance(U_lst, jnp.ndarray | np.ndarray) or U_lst.ndim != 2:
            raise TypeError("U_lst must be a 2D JAX or NumPy array or a list of \
                            2D JAX/NumPy arrays.")
        U_lst = [U_lst]
    else:
        if not all(
            isinstance(U, jnp.ndarray | np.ndarray) and U.ndim == 2 for U in U_lst):
            raise TypeError("All elements in U_lst must be 2D JAX or NumPy arrays.")
        if not all(U.shape == U_lst[0].shape for U in U_lst):
            raise ValueError("All arrays in U_lst must have the same shape.")


    Nu = U_lst[0].shape[1]
    Nt = U_lst[0].shape[0]

    if t is not None:
        if not isinstance(t, jnp.ndarray | np.ndarray) or t.ndim != 1:
            raise TypeError("t must be a 1D JAX or NumPy array.")
        if len(t) != Nt:
            raise ValueError(f"Length of t ({len(t)}) must match the number of time\
                             points in U_lst ({Nt}).")

    if time_series_labels is not None:
        if not isinstance(time_series_labels, list):
            raise TypeError("time_series_labels must be a list of strings.")
        if len(time_series_labels) != len(U_lst):
            raise ValueError(f"Length of time_series_labels ({len(time_series_labels)})\
                             must match the number of time series ({len(U_lst)}).")

    if line_formats is not None:
        if not isinstance(line_formats, list):
            raise TypeError("line_formats must be a list of strings.")
        if len(line_formats) != len(U_lst):
            raise ValueError(f"Length of line_formats ({len(line_formats)}) must \
                             match the number of time series ({len(U_lst)}).")

    if state_var_names is not None:
        if not isinstance(state_var_names, list):
            raise TypeError("state_var_names must be a list of strings.")
        if len(state_var_names) != Nu:
            raise ValueError(f"Length of state_var_names ({len(state_var_names)}) \
                             must match the number of state variables ({Nu}).")

    if t_lim is not None and not isinstance(t_lim, int | float):
         raise TypeError("t_lim must be a number (int or float).")

    # defaults
    plot_kwargs.setdefault('linewidth', 2)

    # setup time vectors
    if t is None:
        t = jnp.arange(Nt)
    if t_lim is None:
        t_lim = t[-1]

    # handle optional inputs
    if time_series_labels is None:
        time_series_labels = [None for _ in range(len(U_lst))]
    if line_formats is None:
        line_formats = ['-' for _ in range(len(U_lst))]

    # plot
    fig, axs = plt.subplots(Nu, figsize = figsize)
    # Ensure axs is always iterable, even if Nu=1
    if Nu == 1:
        axs = [axs]
    for i in range(Nu):
        for j, Y in enumerate(U_lst):
            axs[i].plot(t, Y[:, i], line_formats[j], label=time_series_labels[j],
                         **plot_kwargs)
            axs[i].set_xlim([0, t_lim])
        if state_var_names is not None:
            axs[i].set(ylabel=state_var_names[i])
    if time_series_labels[0] is not None:
        axs[0].legend(loc='upper right')
    axs[-1].set(xlabel=x_label)
    if title is not None:
        axs[0].set_title(title, fontsize=14)
    plt.show()

def imshow_1D_spatiotemp(U,
                         tN,
                         domain=(0,1),
                         figsize=(20, 6),
                         title = None,
                         x_label = r'$t$',
                         **imshow_kwargs):
    """
    Plot 1D spatiotemporal data using imshow.

    Parameters
    ----------
    U: 2D array
        Data to be plotted, shape should be (Nt, Nx) where Nt is the number of time
        points and Nx is the number of spatial points
    tN: float
        Final time of the simulation
    domain: tuple of length 2
        Bounds of the spatial domain, default is (0, 1)
    figsize: tuple
        Size of the figure to be created, default is (20, 6)
    title: string, optional
        Title of the plot, if None no title is shown
    x_label: string, optional
        Label for the x-axis, default is r'$t$'
    **imshow_kwargs: additional arguments to pass to imshow
    """
    # Input validation
    if not isinstance(U, jnp.ndarray | np.ndarray) or U.ndim != 2:
        raise TypeError("U must be a 2D JAX or NumPy array.")
    if not isinstance(domain, tuple) or len(domain) != 2:
        raise TypeError("domain must be a tuple of length 2.")
    if not all(isinstance(x, int | float) for x in domain):
        raise TypeError("Both elements of domain must be numbers (int or float).")

    #set defaults for imshow
    imshow_kwargs.setdefault('aspect', 'auto')
    imshow_kwargs.setdefault('origin', 'lower')
    imshow_kwargs.setdefault('cmap', 'RdGy')
    imshow_kwargs.setdefault('extent', [0, tN, domain[0], domain[1]])

    plt.figure(figsize=figsize, dpi=200)
    plt.imshow(U.T, **imshow_kwargs)
    plt.ylabel('x')
    plt.xlabel(x_label)
    if title is not None:
        plt.title(title)
    plt.colorbar(pad = 0.01, label = r'$u$')
    plt.show()

# TODO: add plot_attrator function to visualize 2D/3D attractors in state space



def plot_in_3D_state_space(U_lst,
                           time_series_labels=None,
                           line_formats=["-", "r--"],
                           state_var_names=None,
                           figsize = (20,8),
                           title = None,
                           **plot_kwargs):
    """ **STEP 1 - clear, one-line summary:** Plot time series data to visualize 3D attractors in state space.

    Parameters
    ----------
    **STEP 2 - 
    <param_name> : <type and optional?>
        <description and shape>**
    U_lst : 2D array or list of 2D arrays
        If a 2D array, shape should be (Nt, 3) with 3 state
        variables, where Nt is the number of time points. If a list of 2D arrays,
        each array should have shape (Nt, 3) and represent different time series.
    time_series_labels : list of strings, optional
        List of strings containing the labels for each time series to be shown in
        a legend. If None, no labels will be shown.
    line_formats : list of strings, optional
        List of strings containing the line formats for each time series. If None,
        default line format will be used.
    state_var_names : list of strings, optional
        List of strings containing the names of the state variables. If None,
        no axis labels will be shown.
    figsize : tuple, optional
        Size of the figure to be created. Default is (20, 8).
    title : string, optional
        Title of the plot. If None, no title is shown.
    plot_kwargs : dict, optional
        Additional arguments to pass to the plot function.
    """

    # **STEP 3** 
    # Input validation
    if not isinstance(U_lst, list):
        if not isinstance(U_lst, jnp.ndarray | np.ndarray) or U_lst.ndim != 2:
            raise TypeError("U_lst must be a 2D JAX or NumPy array or a list of \
                            2D JAX/NumPy arrays.")
        U_lst = [U_lst]
    else:
        if not all(
            isinstance(U, jnp.ndarray | np.ndarray) and U.ndim == 2 for U in U_lst):
            raise TypeError("All elements in U_lst must be 2D JAX or NumPy arrays.")
        if not all(U.shape == U_lst[0].shape for U in U_lst):
            raise ValueError("All arrays in U_lst must have the same shape.")
        
    Nu = U_lst[0].shape[1]

    if time_series_labels is not None:
        if not isinstance(time_series_labels, list):
            raise TypeError("time_series_labels must be a list of strings.")
        if len(time_series_labels) != len(U_lst):
            raise ValueError(f"Length of time_series_labels ({len(time_series_labels)})\
                             must match the number of time series ({len(U_lst)}).")

    if line_formats is not None:
        if not isinstance(line_formats, list):
            raise TypeError("line_formats must be a list of strings.")
        if len(line_formats) != len(U_lst):
            raise ValueError(f"Length of line_formats ({len(line_formats)}) must \
                             match the number of time series ({len(U_lst)}).")

    if state_var_names is not None:
        if not isinstance(state_var_names, list):
            raise TypeError("state_var_names must be a list of strings.")
        if len(state_var_names) != Nu:
            raise ValueError(f"Length of state_var_names ({len(state_var_names)}) \
                             must match the number of state variables ({Nu}).")

    # **STEP 4** 
    # defaults
    plot_kwargs.setdefault('linewidth', 2)

    # **STEP 5** 
    # handle optional inputs
    if time_series_labels is None:
        time_series_labels = [None for _ in range(len(U_lst))]
    if line_formats is None:
        line_formats = ['-' for _ in range(len(U_lst))]

    # **STEP 5 - CODE:** 
    # plot
    fig, axs = plt.subplots(subplot_kw={"projection": "3d"}, figsize = figsize)
    # Ensure axs is always iterable, even if Nu=1
    if Nu == 1:
        axs = [axs]
    for j, Y in enumerate(U_lst):
        axs.plot(Y[:, 0], Y[:, 1], Y[:, 2], line_formats[j], 
                        label=time_series_labels[j],
                        **plot_kwargs)
    if state_var_names is not None:
        axs.set(xlabel=state_var_names[0])
        axs.set(ylabel=state_var_names[1])
        axs.set(zlabel=state_var_names[2])
    if time_series_labels[0] is not None:
        axs.legend(loc='upper right')
    if title is not None:
        axs.set_title(title, fontsize=14)
    plt.show()