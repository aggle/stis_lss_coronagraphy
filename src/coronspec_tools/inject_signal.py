"""
Tools for injecting spectra into 2-D spectral images
"""
from pathlib import Path

import numpy as np

from astropy import units
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.modeling.models import Gaussian1D

from coronspec_tools import utils as ctutils
from coronspec_tools import observing_sequence

class Injector(observing_sequence.ObsSeq):

    def inject_template_at_row(
            self,
            row_ind : int,
            template_spectrum : np.ndarray,
            scale : float = 1
    ):
        """
        Inject the template spectrum at a given row
        """
        inj_img = self.occ_img.copy()
        width = np.floor(self.unocc_trace.shape[0]/2).astype(int)
        bottom = row_ind - width
        top = row_ind + width + 1

        # transform the unocculted spectrum to the given spectrum
        template = self.unocc_trace.data * (template_spectrum / self.primary_spectrum)
        template *= scale
        
        inj_img[bottom:top] += scale*self.unocc_trace.data




def inject_template_at_row(
    shape : tuple,
    row_ind : int,
    template_spectrum : np.ndarray,
    scale : float = 1
):
    """
    Inject the template spectrum at a given row
    """
    img = np.zeros(shape, dtype=float)
    width = np.floor(template_spectrum.shape[0]/2).astype(int)
    bottom = row_ind - width
    top = row_ind + width + 1
    # transform the unocculted spectrum to the given spectrum
    # template = self.unocc_trace * (template_spectrum / self.primary_spectrum)
    # template *= scale
    img[bottom:top] += template_spectrum
    return img

def make_gaussian_signal(nrows, fwhm, spectrum):
    center = (nrows-1)//2
    ncols = spectrum.size
    signal = np.tile(Gaussian1D(1, center, fwhm/2.35)(np.arange(nrows))[:, None], (1, ncols)) * spectrum
    return signal
