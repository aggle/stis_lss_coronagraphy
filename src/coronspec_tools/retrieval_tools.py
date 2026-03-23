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
        self.template_array = sdi.obs.occ_stamp.data.copy()
        self.template_trace = self.flatten_unocc_trace()

    def compute_throughput_map(self):
        """
        add a normalized signal at each row to an empty array and run the scaling-subtracting-descaling algorithm
        """
        pass

    def flatten_unocc_trace(self, zero_max : bool = False) -> np.ndarray:
        """
        For the non-rectified images, the trace usually isn't straight.
        Straighten it out before you use it for injection

        Parameters
        ----------
        zero_max : bool = False
          Zero an outlier pixel. If True, the largest pixel is set to 0.
          This argument is particular to the HD-283593 dataset.

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
        Renormalize the trace to have the shape of the given input spectrum, while preserving the total flux?
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

    def crosscorr(self, data : np.ndarray, model : np.ndarray) -> float:
        """
        Compute the cross-correlation of the signal with the forward-modeled injection spectrum

        Parameters
        ----------
        data : np.ndarray
          1-d array that may contain signal
        model : np.ndarray
          1-d array of a model that may be in the data

        Output
        ------
        cc : float
          the dot product of data and model (mean-subtracted)
        """
        data = np.array(data)
        model = np.array(model)
        # mask nans 
        wherenan = np.isnan(data) | np.isnan(model)
        data = data[~wherenan].copy()
        model = model[~wherenan].copy()
        cc = np.dot(
            data - np.mean(data),
            model - np.mean(model)
        )/(data.size * model.size)**0.5
        return cc

    def inject_and_process(
        self,
        template_img : np.ndarray | None,
        inj_row : int,
        template_trace : np.ndarray | None,
        spectrum : np.ndarray,
        scale : float,
    ):
        """
        Wrapper to:
        1. reshape the spectrum to the desired shape
        2. set the flux scale,
        3. create an image with the trace injected
        """
        if template_img is None:
            template_img = self.template_array.copy()
        if template_trace is None:
            template_trace = self.template_trace
        trace = self.renormalize_trace(template_trace, spectrum, scale)
        trace = template_trace * scale
        inj_template = self.add_trace_to_template(trace, inj_row, template_img)
        self.inj_trace = trace
        self.inj_img = inj_template
        # create an SDI instance with the injected signal
        self.inj_sdi = sdi_tools.SDI(
            obs = self.obs,
            ref_wl_ind = self.sdi.ref_wl_ind,
            psf_halfwidth = self.sdi.psf_halfwidth,
            stamp_to_subtract = self.inj_img
        )
        self.inj_sdi.compute_scaled_stamp(
            stamp=inj_template, stamp_center = self.obs.occ_stamp_center
        )
        self.inj_sdi.generate_model_results_df(inj_row, inj_row)
        # extract the signal in the given row by descaling the same
        self.inj_sdi.model_results['signal'] = self.inj_sdi.model_results.apply(
            lambda row: self.inj_sdi.descale_trace(
                row['residual'], row['trace'], row['row_indices'][[0, -1]]
            ),
            axis=1
        )
        # get the shape of the expected signal by applying the PSF model to the template PSF
        self.inj_sdi.model_results['fm_injection'] = self.inj_sdi.model_results.apply(
            lambda row: self.inj_trace[np.floor(self.inj_trace.shape[0]/2).astype(int)] - row['model_descaled'],
            axis=1
        )
        # compute the correlation between the residual and the forward-modeled signal
        self.inj_sdi.model_results['fm_ccorr'] = self.inj_sdi.model_results.apply(
            lambda row: self.crosscorr(row['signal'], row['fm_injection']),
            axis=1
        )
        # compute the correlation between the residual and the data without injection
        self.inj_sdi.model_results['fm_ccorr_nosignal'] = self.inj_sdi.model_results.apply(
            lambda row: self.crosscorr(self.obs.occ_stamp.data[row.name], row['fm_injection']),
            axis=1
        )
        self.inj_results = self.inj_sdi.model_results.copy()




