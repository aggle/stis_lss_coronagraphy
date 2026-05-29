"""
Handy catch-all for diagnostic plots
"""
import matplotlib as mpl
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

from coronspec_tools import (
    observing_sequence,
    sdi_tools,
    retrieval_tools
)


def plot_injection_results(
        ret: retrieval_tools.Retriever,
        row : pd.Series
) -> mpl.figure.Figure:
    """
    Plot the results of injecting and retrieving a spectrum

    Parameters
    ----------
    ret: retrieval_tools.Retriever
      a Retriever instance
    row : pd.Series
      a row of ret.inj_results e.g. ret.inj_results.loc[21]
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
    ax.set_title("Recovered signal [single row]")
    trace_center = np.floor(ret.inj_trace.shape[0]/2).astype(int)
    inj_spec = ret.inj_trace[trace_center]
    ax.plot(inj_spec, label='injected', c='C0')
    ax.plot(row['signal'], label='recovered', c='C1')
    ax.plot(row['model_descaled'], label='psf model', c='C2')
    ax.plot(row['fm_injection'], label='FM injection', c='C3')
    ax.legend()

    return fig


def plot_row_fitting(
        results_row : pd.Series,
) -> mpl.figure.Figure:
    residual_img = results_row['residual']

    scaled_ind = results_row['row_indices']
    naxes = len(scaled_ind)
    ncols = np.ceil(np.sqrt(naxes)).astype(int)
    nrows = np.ceil(naxes/ncols).astype(int)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows),
        sharex=True, sharey=False,
    )

    cols = np.arange(residual_img.shape[1])

    for i, (ax, scaled_row_ind) in enumerate(zip(axes.flat, scaled_ind[::-1])):
        i = scaled_ind.size - 1 - i
        ax.set_title(f"Scaled row: {scaled_row_ind}")
        mask = results_row['stamp_mask'][i]
        scaled_row = results_row['scaled_stamp'][i]
        scaled_unc = results_row['scaled_stamp_unc'][i]
        model = results_row['model'][i]
        ax.errorbar(
            cols,
            np.ma.masked_array(scaled_row, mask),
            # yerr = np.ma.masked_array(scaled_unc, mask),
            c='C0'
        )
        ax.errorbar(
            cols,
            np.ma.masked_array(scaled_row, ~mask),
            # yerr = np.ma.masked_array(scaled_unc, ~mask),
            c='C1'
        )
        ax.plot(cols, np.ma.masked_array(model, mask), c='C2')

    return fig


def plot_row_fitting_hist(
    results_row : pd.Series,
) -> mpl.figure.Figure:
    residual_img = results_row['residual']

    scaled_ind = results_row['row_indices']
    naxes = len(scaled_ind)
    ncols = np.ceil(np.sqrt(naxes)).astype(int)
    nrows = np.ceil(naxes/ncols).astype(int)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows),
        sharex=False, sharey=False
    )

    cols = np.arange(residual_img.shape[1])

    for i, (ax, scaled_row_ind) in enumerate(zip(axes.flat, scaled_ind)):
        ax.set_title(f"Scaled row: {scaled_row_ind}")
        mask = results_row['stamp_mask'][i]
        residual = results_row['residual'][i][~mask]
        ax.hist(
            residual,
            bins=int(residual.size/10),
            log=False,
            histtype='step',
        )
        ax.axvline(0, c='k', alpha=0.5)
    for ax in axes.flat:
        ax.set_ylabel("Npix")
        ax.set_xlabel("Residual")
        ax.label_outer()
    return fig
