"""
Tools for reconstructing spectra from PSF-subtracted images
"""
from pathlib import Path

import numpy as np
from scipy import ndimage

import matplotlib as mpl
from matplotlib import pyplot as plt

from astropy import units
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.modeling.models import Gaussian1D

from coronspec_tools import utils as ctutils
from coronspec_tools import observing_sequence
from coronspec_tools import sdi_tools

class Retriever:
    def __init__(self, sdi:sdi_tools.SDI):
        self.sdi = sdi
        self.obs = sdi.obs
        self.template_array = sdi.obs.occ_stamp.data.copy()
        self.template_trace = self.flatten_unocc_trace()

    def compute_throughput_map(self):
        """
        add a normalized signal at each row to an empty array and run the scaling-subtracting-descaling algorithm
        """
        pass

    def flatten_unocc_trace(self, zero_max : bool = False) -> np.ndarray:
        """
        For the non-rectified images, the trace usually isn't straight. Straighten it out before you use it for injection
        """
        pad = 3
        width = self.obs.unocc_trace.data.shape[0] + 2*pad
        new_trace = self.obs.get_unocc_trace(width).data.copy()
        if zero_max:
            new_trace.flat[new_trace.argmax()] = 0
        cols = np.arange(new_trace.shape[1])
        line_func = np.polynomial.Polynomial.fit(
            cols, new_trace.argmax(axis=0),
            1
        )
        centers = line_func(cols)
        trace_center = np.floor(new_trace.shape[0]/2).astype(int)
        col_shifts = trace_center - centers
        shifted_cols = []
        for col, shift in zip(new_trace.T, col_shifts):
            shifted_cols.append(ndimage.shift(col, shift, mode='mirror'))
        flat_trace = np.stack(shifted_cols).T[pad:-pad]
        # normalize each column
        flat_trace = flat_trace/flat_trace.sum(axis=0)
        # normalize the flux to the unocculted trace flux
        unocc_norm = self.obs.unocc_trace.data.sum()
        flat_norm = flat_trace.sum()
        flat_trace *= unocc_norm/flat_norm

        # # shift and scale to match the original trace
        # offset = self.obs.unocc_trace.data.mean() - flat_trace.mean()
        # scale = np.ptp(self.obs.unocc_trace.data)/np.ptp(flat_trace)
        # flat_trace = (flat_trace-flat_trace.mean()) * scale + offset
        return flat_trace

    def renormalize_trace(
            self,
            trace : np.ndarray,
            spectrum : np.ndarray,
            scale : float = 1.
    ):
        """
        Renormalize the trace to have the shape of the spectrum, while preserving the total flux?
        Spectrum must have same units as self.obs.primary.spectrum_flux
        """
        # convert spectrum to counts
        # spectrum /= self.obs.throughput_corr.value
        norm = spectrum #/ self.obs.primary_spectrum_flux.value
        renormalized_trace = trace * norm * scale
        return renormalized_trace

    def add_trace_to_template(
            self, trace, inj_row, template : np.ndarray | None = None
    ) -> np.ndarray:
        # pad trace with zeros to match shape
        if template is None:
            template = self.template_array.copy()
        halfwidth = int((trace.shape[0] - trace.shape[0]%2)/2)
        lb = inj_row - halfwidth
        ub = lb + trace.shape[0]
        trace_trim = [0, trace.shape[0]]
        if lb < 0:
            trace_trim[0] = -lb 
            lb = 0
        if ub > template.shape[0]:
            trace_trim[1] -= ub - template.shape[0]
            ub = template.shape[0]
        template[lb:ub] = template[lb:ub] + trace[trace_trim[0]:trace_trim[1]]
        return template

    def inject_and_process(
        self,
        template_img : np.ndarray | None,
        inj_row : int,
        template_trace : np.ndarray | None,
        spectrum : np.ndarray,
        scale : float,
    ):
        if template_img is None:
            template_img = self.template_array.copy()
        if template_trace is None:
            template_trace = self.template_trace
        trace = self.renormalize_trace(template_trace, spectrum, scale)
        template = self.add_trace_to_template(trace, inj_row, template_img)
        self.inj_trace = trace
        self.inj_img = template
        injsdi = sdi_tools.SDI(self.obs, self.sdi.ref_wl_ind, self.sdi.psf_halfwidth)
        injsdi.compute_scaled_stamp(
            stamp=template, stamp_center = self.obs.occ_stamp_center
        )
        injsdi.generate_model_results_df(inj_row, inj_row)
        injsdi.model_results['signal'] = injsdi.model_results.apply(
            lambda row: injsdi.descale_trace(
                row['residual'], row['trace'], row['row_indices'][[0, -1]]
            ),
            axis=1
        )
        self.inj_results = injsdi.model_results.copy()




    def plot_results(self, row):
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
        ax.plot(self.inj_trace.max(axis=0), label='injected')
        ax.plot(row['signal'], label='recovered')
        ax.legend()

        return fig
