"""
Tools for finding the position of the star in the occulted (or unocculted!) data
"""

from pathlib import Path
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy import units
from astropy.wcs import WCS

# Unocculted star finding
from scipy import interpolate
from scipy import optimize

from coronspec_tools import utils as ctutils


def interp_col(col):
    x = np.arange(col.size)
    func = interpolate.make_interp_spline(x, col)
    return func

def interp_peak(
        hdulist : fits.HDUList,
        search_col : int = 200,
) -> float :
    """
    Guess the position of the star

    Parameters
    ----------
    hdulist : fits.HDUList,
      HDUList with the data in the "SCI" header
    search_col : int = 200,

    Output
    ------
    Define your output

    """
    pass # insert body here
    is_str = isinstance(hdulist, str)
    if is_str:
        hdulist = fits.open(hdulist)
    # initial guess
    star_row = float(hdulist['SCI'].header['CRPIX2'])-1
    img = hdulist['SCI'].data

    interp_rows = np.arange(star_row-20, star_row+20+1, dtype=int)
    func = interpolate.interp1d(interp_rows, -img[interp_rows, search_col], 'cubic')
    interp_min = optimize.minimize_scalar(func, (float(interp_rows[0]), star_row, float(interp_rows[-1])))
    peak_row = interp_min.x
    if is_str:
        hdulist.close()
    return peak_row

def find_unocc_pos(hdulist, wlsol=None):
    """Find the position in degrees"""
    img = hdulist[1].data
    wcs = WCS(hdulist[1].header)
    # generate the wavelength solution along the nominal star position
    if wlsol is None:
        wlsol = wcs.pixel_to_world(
            np.arange(img.size),
            np.ones(img.size)*wcs.wcs.crpix[1]
        )
    peak_wl, peak_offset = [], []
    for wl_index in np.arange(5, 201, 10):
        wl = wlsol[wl_index] 
        peak_col = wcs.world_to_pixel_values(
            wl,
            0*units.deg,
        )[0]
        peak_row = interp_peak(hdulist, int(peak_col))
        offset = wcs.pixel_to_world(
            peak_col, peak_row
        )[1]
        peak_wl.append(wl)
        peak_offset.append(offset)
    # fit a polynomial to the peak positions
    peak_wl, peak_offset = units.Quantity(peak_wl), units.Quantity(peak_offset)
    polyfit = np.polynomial.Polynomial.fit(peak_wl.value, peak_offset.value, 3)
    meas_offset = polyfit(wlsol[0].value)*units.deg
    meas_row = wcs.world_to_pixel(wlsol[0], meas_offset)[1]
    return meas_offset, meas_row


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
        meas_offset = find_unocc_pos(hdulist, wavelengths)[0]
    # apply the shift to the occulted exposure
    with fits.open(occ_2d_file) as hdulist:
        postarg2 = (hdulist[0].header['POSTARG2'] * units.arcsec).to(units.deg)
        wcs = WCS(hdulist[1].header)
        # occ_col, occ_row = w.world_to_pixel_values(
        #     wl_lo, 
        #     units.Quantity(0, unit='deg')
        # )
        # occ_pos = occ_row - shift
        occ_pos = wcs.world_to_pixel(
            wl_lo,
            meas_offset + postarg2
        )
    if return_offset:
        return occ_pos, meas_offset
    else:
        return occ_pos


