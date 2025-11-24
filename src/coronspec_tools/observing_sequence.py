"""
Tools for managing an observing sequence
"""
from pathlib import Path

import numpy as np

from astropy import units
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

from coronspec_tools import utils as ctutils
from coronspec_tools import find_star as ctfs

class ObsSeq:

    def __init__(
            self,
            sx1_file : str | Path,
            unocc_file : str | Path,
            occ_file : str | Path,
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

        Output
        ------
        None

        """
        self._files = {'sx1': sx1_file, 'unocc': unocc_file, 'occ': occ_file}
        # spectral information
        with fits.open(sx1_file) as hdulist:
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
        with fits.open(unocc_file) as hdulist:
            self.unocc_wcs = WCS(hdulist[1].header)
            self.unocc_img = hdulist[1].data.copy()
            self.offset, self.unocc_row = ctutils.find_unocc_pos(hdulist, self.wlsol)
            self.unocc_trace = Cutout2D(
                self.unocc_img, 
                position=(self.unocc_img.shape[1]/2, self.unocc_row),
                size=(11, self.unocc_img.shape[1]),
                wcs=self.unocc_wcs
            )
        with fits.open(occ_file) as hdulist:
            self.occ_wcs = WCS(hdulist[1].header)
            self.occ_img = hdulist[1].data.copy()
            (occ_col, occ_row) = ctfs.find_star_from_wcs(
                sx1_file, unocc_file, occ_file,
            )
            self.occ_row = occ_row
        return None
