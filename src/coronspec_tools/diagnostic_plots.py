"""
Handy catch-all for diangostic plots
"""
import matplotlib as mpl
from matplotlib import pyplot as plt

import numpy as np

from coronspec_tools import (
    observing_sequence,
    sdi_tools,
    retrieval_tools
)


def plot_injection_results(
        ret: retrieval_tools.Retriever,
        row : retrieval_tools.pd.Series
):
    """
    Plot the results of injecting and retrieving a spectrum
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    trace = row['trace']
    xcoords = np.arange(-0.5, trace.size+0.5)
    ycoords = np.arange(*(row['row_indices'][[0, -1]]+np.array([-0.5, 1.5])))

    ax = axes[0, 0]
    ax.set_title("Scaled stamp")
    imax = ax.pcolormesh(xcoords, ycoords, row['scaled_stamp'])
    fig.colorbar(imax, ax=ax)

    ax = axes[0, 1]
    ax.set_title("Model")
    imax = ax.pcolormesh(xcoords, ycoords, row['model'])
    fig.colorbar(imax, ax=ax)

    ax = axes[1, 0]
    ax.set_title("Residual")
    imax = ax.pcolormesh(xcoords, ycoords, row['residual'])
    fig.colorbar(imax, ax=ax)

    for ax in axes.flat[:3]:
        ax.plot(row['trace'], ls='--', c='gray')

    ax = axes[1, 1]
    ax.set_title("Recovered signal")
    ax.plot(ret.inj_trace.max(axis=0), label='injected')
    ax.plot(row['signal'], label='recovered')
    ax.legend()

    return fig

