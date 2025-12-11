"""
Tools for reconstructing spectra from PSF-subtracted images
"""
from pathlib import Path

import numpy as np
from scipy import ndimage

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

    def reproject_signal(self, residual_img, y_vals):
        """
        residual_img : np.ndarray
          the residual, wavelength-scaled image
        y_vals : np.ndarray
          the y value in each column for which to estimate the signal
        """
        cols = np.arange(residual_img.shape[1])
        signal = np.zeros_like(cols)*np.nan
        for c in cols:
            r = y_vals[c]
            r_lo, r_hi = [f(r).astype(int) for f in (np.floor, np.ceil)]
            weights = np.abs(r-r_lo)**-2, np.abs(r_hi-r)**-2
            signal[c] = np.sum(residual_img[[r_lo,r_hi], c]*weights) / np.sum(weights)
        return signal

    def compute_throughput_map(self):
        """
        add a normalized signal at each row to an empty array and run the scaling-subtracting-descaling algorithm
        """
        pass

    def flatten_trace(self) -> np.ndarray:
        """
        For the non-rectified images, the trace usually isn't straight. Straighten it out before you use it for injection
        """
        pad = 4
        width = self.obs.unocc_trace.data.shape[0] + 2*pad
        new_trace = self.obs.get_unocc_trace(width).data.copy()
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
        # shift and scale to match the original trace
        offset = self.obs.unocc_trace.data.mean() - flat_trace.mean()
        scale = np.ptp(self.obs.unocc_trace.data)/np.ptp(flat_trace)
        flat_trace = (flat_trace-flat_trace.mean()) * scale + offset
        return flat_trace
