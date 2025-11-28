"""
Tools for injecting spectra into 2-D spectral images
"""
from pathlib import Path

import numpy as np

from astropy import units
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

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
        template = self.unocc_trace * (template_spectrum / self.primary_spectrum)
        template *= scale
        
        inj_img[bottom:top] += scale*self.unocc_trace


