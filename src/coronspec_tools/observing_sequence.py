"""
Tools for managing an observing sequence
"""
from pathlib import Path
import warnings

import numpy as np

from astropy import units
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS, FITSFixedWarning
from astropy.utils.exceptions import AstropyWarning

from coronspec_tools import utils as ctutils
from coronspec_tools import find_star as ctfs

# the WCS throws an warning when loading this data that we can ignore
warnings.filterwarnings("ignore", category=FITSFixedWarning)

class ObsSeq:

    def __init__(
            self,
            sx1_file : str | Path,
            unocc_file : str | Path,
            occ_file : str | Path,
            trace_width : int = 11,
            occ_stamp_width : int = 61,
            contrast : bool = True,
            median_clean : int = 10
    ) -> None :
        """
        Instantiate a class to manage injection of spectral traces into 2-D
        spectral images. This includes rescaling and reshaping the spectra.

        Parameters
        ----------
        sx1_file : str | Path
          file containing the wavelength solution and extracted spectrum
        unocc_file : str | Path
          file containing the 2-D unocculted spectral image
        occ_file : str | Path
          file containing the 2-D occulted spectral image
        trace_width : int = 11
          the width of the trace image to cut out, for PSF injection
        occ_stamp_width : int = 61
          the width of the occulted image to cut out for PSF subtraction
        contrast : bool = True
          if True, convert the occ and unocc images to counts/sec, and divide
          by the primary unocculted spectrum
        median_clean : int = 10
          Apply a rolling median along the wavelength axis with this width to
          clean the data. If >= 0, not applied.

        Output
        ------
        None

        """
        self._files = {'sx1': sx1_file, 'unocc': unocc_file, 'occ': occ_file}
        self.hdrs = {
            k: {h: fits.getheader(v, h) for h in [0,'sci']}
            for k, v in self._files.items()
        }
        # pull data out of the files
        # spectral information
        self.process_specfile(sx1_file)
        # unocculted
        self.process_unocculted(unocc_file, trace_width)
        # unocculted
        self.process_occulted(occ_file)

        # process data
        (occ_col, occ_row) = ctfs.find_star_from_wcs(
            sx1_file, unocc_file, occ_file,
        )
        self.occ_row = occ_row
        self.occ_stamp = Cutout2D(
            self.occ_img, 
            position=(self.occ_img.shape[1]/2, self.occ_row),
            size=(occ_stamp_width, self.occ_img.shape[1]),
            wcs=self.occ_wcs
        )
        self.occ_stamp_center = self.occ_row - self.occ_stamp.origin_original[1]
        # data cleaning
        if median_clean > 0:
            specunit = self.primary_spectrum.unit
            self.primary_spectrum = ctutils.rolling_median(
                self.primary_spectrum.value, 10
            ) * specunit
            self.primary_spectrum_unc = ctutils.rolling_median(
                self.primary_spectrum_unc.value, 10
            ) * specunit
            self.occ_stamp.data = self.clean_stamp(self.occ_stamp.data, 10)
        if contrast:
            # convert to units of contrast
            self.convert_to_contrast()
        return None

    def process_specfile(self, specfile):
        with fits.open(specfile) as hdulist:
            table = hdulist[1].data
            colname = 'WAVELENGTH'
            colind = table.names.index(colname)+1
            colunit = hdulist[1].header[f'TUNIT{colind}']
            if colunit == 'Angstroms':
                colunit = 'Angstrom'
            self.wlsol = units.Quantity(
                np.squeeze(table[colname]),
                unit=colunit
            ).to(units.m)
            colname = 'NET'
            colind = table.names.index(colname)+1
            colunit = hdulist[1].header[f'TUNIT{colind}']
            if 'Counts' in colunit:
                colunit = colunit.replace("Counts",'count')
            self.primary_spectrum = units.Quantity(
                np.squeeze(table[colname]),
                unit=colunit
            )
            colname = 'NET_ERROR'
            colind = table.names.index(colname)+1
            colunit = hdulist[1].header[f'TUNIT{colind}']
            if 'Counts' in colunit:
                colunit = colunit.replace("Counts",'count')
            self.primary_spectrum_unc = units.Quantity(
                np.squeeze(table[colname]),
                unit=colunit
            )
            colname = 'FLUX'
            colind = table.names.index(colname)+1
            colunit = hdulist[1].header[f'TUNIT{colind}']
            if 'Counts' in colunit:
                colunit = colunit.replace("Counts",'count')
            self.primary_spectrum_flux = units.Quantity(
                np.squeeze(table[colname]),
                unit=colunit
            )
            colname = 'ERROR'
            colind = table.names.index(colname)+1
            colunit = hdulist[1].header[f'TUNIT{colind}']
            if 'Counts' in colunit:
                colunit = colunit.replace("Counts",'count')
            self.primary_spectrum_flux_unc = units.Quantity(
                np.squeeze(table[colname]),
                unit=colunit
            )
        return

    def process_unocculted(self, unocc_file, trace_width):
        # measure the TA offset
        # get the wcs and unocculted image
        with fits.open(unocc_file) as hdulist:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', AstropyWarning)
                self.unocc_wcs = WCS(hdulist[1].header)
            self.unocc_img = hdulist[1].data.copy()
            self.offset, self.unocc_row = ctfs.find_unocc_pos(hdulist, self.wlsol)
        self.unocc_trace = self.get_unocc_trace(trace_width)
        return

    def process_occulted(self, occ_file):
        # get the wcs and occulted image
        with fits.open(occ_file) as hdulist:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.occ_wcs = WCS(hdulist[1].header)
            self.occ_img = hdulist[1].data.copy()

    def get_cutout(self, row_ind):
        """Return a cutout from the occulted exposure"""
        return Cutout2D(
            self.occ_img, 
            position=(self.occ_img.shape[1]/2, row_ind),
            size=(101, self.occ_img.shape[1]),
            wcs=self.occ_wcs
        )

    def clean_stamp(self, img, width=10):
        """Apply median filtering to a 2-D spectral image"""
        return ctutils.median_filter_image(img, width)


    def get_unocc_trace(self, trace_width):
        trace = Cutout2D(
            self.unocc_img, 
            position=(self.unocc_img.shape[1]/2, self.unocc_row),
            size=(trace_width, self.unocc_img.shape[1]),
            wcs=self.unocc_wcs
        )
        return trace

    def convert_to_contrast(self):
        """Converts all relevant data to units of contrast"""
        exptime = self.hdrs['unocc']['sci']['exptime']
        self.unocc_img = self.unocc_img / exptime
        self.unocc_img = self.unocc_img / self.primary_spectrum.value
        self.unocc_trace.data = self.unocc_trace.data / self.primary_spectrum.value
        exptime = self.hdrs['occ']['sci']['exptime']
        self.occ_img = self.occ_img / exptime
        self.occ_img = self.occ_img / self.primary_spectrum.value
        self.occ_stamp.data = self.occ_stamp.data / self.primary_spectrum.value
        return
