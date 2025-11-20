"""
Tools for finding the position of the star in the occulted (or unocculted!) data
"""

from pathlib import Path
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy import units
from astropy import wcs

from coronspec_tools import utils as ctutils


# Unocculted star finding
from scipy import interpolate
from scipy import optimize

def interp_col(col):
    x = np.arange(col.size)
    func = interpolate.make_interp_spline(x, col)
    return func

def find_unocc_peaks(
        unocc_img,
        row_range : tuple | None = None,
        col_mask : np.ndarray | None = None,
) -> pd.Series :
    """
    Find the column-wise peaks of an unocculted spectrum.

    Parameters
    ----------
    unocc_img : np.ndarray
      the 2-D image to fit. can be flt or sx2.
    row_range : tuple(lo, hi)
      restrict the search range for the peak to these rows
    col_mask: np.ndarray
      an int array of indexes; only search these columns for the peak

    Output
    ------
    peaks : pd.Series
      indexed by column; the row value where the peak is found

    """
    if col_mask is None:
        col_mask = np.arange(unocc_img.shape[1])
    models = {i: interp_col(unocc_img[:, i]) for i in col_mask}
    if row_range is None:
        lo = 0
        hi = unocc_img.shape[0]
    else:
        lo, hi = row_range
    peaks = pd.Series({
        col: optimize.minimize_scalar(lambda x: 1/(model(x)**2), bounds=[lo, hi]).x for col, model in models.items()
    })
    return peaks


def find_star_from_wcs(
        sx1_file : str | Path,
        unocc_2d_file : str | Path,
        occ_2d_file : str | Path,
        return_offset : bool = False,
) -> tuple[float, float] | tuple[tuple, float]:
    """
    Use the WCS to find the pixel coordinates of the occulted star. The WCS has
    units of angular separation on one axis, with the target being placed at 0,
    and wavelength on the other axis. We will find the position of the star in
    (wl, deg) space aqnd convert that to pixels in the occulted file. For this
    we need three files: 1. sx1_file : this tells us the wavelengths present in
    the image 2. sx2_unocc_file : the unocculted sx2 exposure. We use this to
    measure the offset of the actual trace, in separation, from the nominal
    position. 3. sx3_occ_file : the occulted sx2 exposure. We apply the offset
    measured in the unocculted file to the nominal position reported in this
    file.

    Parameters
    ----------
    sx1_file : str | Path,
      The extraceted, unocculted spectrum, We want the wavelength solution.
    unocc_2d_file : str | Path
      The unocculted 2-D spectral image. We want to find the peaks of the PSF along the columns.
    occ_2d_file : str | Path
      The occulted 2-D spectral image. We need the WCS header.
    return_offset : bool = False
      if True, also return the difference (in degrees) between the nominal and measured positions

    Output
    ------
    Returns the row position of the star in the occulted exposure. If
    return_offset is True, then it returns a tuple where the second element is
    the distance in degrees from the nominal position.

    """
    # get the wavelengths of the spectrum
    with fits.open(sx1_file) as hdulist:
        colname = hdulist[1].header['TTYPE3']
        colunit = hdulist[1].header['TUNIT3']
        if colunit == 'Angstroms':
            colunit = 'Angstrom'
        wavelengths = units.Quantity(
            np.squeeze(hdulist[1].data[colname]),
            unit=colunit
        ).to(units.m)
    wl_lo = wavelengths.min()
    # measure the shift in nominal vs actual position in the unocc exposure
    with fits.open(unocc_2d_file) as hdulist:
        w = wcs.WCS(hdulist[1].header)
        nominal_col, nominal_row = w.world_to_pixel_values(
            wl_lo, 
            units.Quantity(0, unit='deg')
        )
        # temporary - measure the position from a single column
        # define the column at which to measure the position
        peak_wl, peak_offset = [], []
        for wl_index in np.arange(5, 51, 5):
            wl =wavelengths[wl_index] 
            peak_col = w.world_to_pixel_values(
                wl,
                0*units.deg,
            )[0]
            peak_row = ctutils.interp_peak(hdulist, int(peak_col))
            offset = w.pixel_to_world(
                peak_col, peak_row
            )[1]
            peak_wl.append(wl)
            peak_offset.append(offset)
        # fit a line to the peak positions
        peak_wl, peak_offset = units.Quantity(peak_wl), units.Quantity(peak_offset)
        line = np.polynomial.Polynomial.fit(peak_wl.value, peak_offset.value, 1)
        # fit_pos = 
        meas_offset = line(wl_lo.value)*units.deg
        # shift = nominal_row - peak_row
    # apply the shift to the occulted exposure
    with fits.open(occ_2d_file) as hdulist:
        postarg2 = (hdulist[0].header['POSTARG2'] * units.arcsec).to(units.deg)
        w = wcs.WCS(hdulist[1].header)
        # occ_col, occ_row = w.world_to_pixel_values(
        #     wl_lo, 
        #     units.Quantity(0, unit='deg')
        # )
        # occ_pos = occ_row - shift
        occ_pos = w.world_to_pixel(
            wl_lo,
            meas_offset + postarg2
        )
    if return_offset:
        return occ_pos, meas_offset
    else:
        return occ_pos
