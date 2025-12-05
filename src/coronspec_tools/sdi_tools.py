import numpy as np
from scipy import ndimage
from scipy import interpolate
import pyklip

def calc_scaling(
        wlmap, refwv_ind : int = 500
) -> np.ndarray:
    """
    compute the rescaling factors for each wavelength slice

    Parameters
    ----------
    wlmap : np.ndarray
      2-D map the same size as the image, with the wavelength corresponding to
      each pixel
    refwv_ind : int = 500

    Output
    ------
    Define your output

    """
    refwl = wlmap[:, refwv_ind]
    rescaled = wlmap.T/refwl
    return rescaled.T

def rescale_img(
    img : np.ndarray,
    center_row : float,
    scale_factors : np.ndarray
) -> np.ndarray:
    """
    compute the rescaling factors for each wavelength slice

    Parameters
    ----------
    img : np.ndarray
      2-D spectral image. Rows are separation, Cols are wavelength
    center_row : float
      the position of the star in the image
    scale_factors : np.ndarray
      how much to scale each position. essentially, the wavelength solution normalized to some index.

    Output
    ------
    scaled_img : np.ndarray
      the image rescale to some wavelength index
    """
    row_coords, col_coords = np.mgrid[:img.shape[0], :img.shape[1]]
    row_sep = row_coords - center_row
    new_rows = (row_sep * scale_factors) + center_row
    scaled_img = ndimage.map_coordinates(img.copy(), [new_rows, col_coords], mode='nearest')
    return scaled_img




def compute_wl_mask_center(
    y : int,
	y0 : float,
	ref_wl_ind : float,
	psf_width : float
) -> float:
    """
    Compute the width of the mask, along the wavelength axis, required to
    mask out a rescaled companion PSF.

    Parameters
    ----------
    y : int
      the row you are interpolating
	y0 : float
      the central row from which the scaling is computed
	ref_wl_ind : float
      the index of the reference wavelength for the scaling
	psf_width : float
      the spatial (y) half-size of the psf

    Output
    ------
    mask_width : float
      the mask-width in wavelength space required to mask out the spatially-rescaled PSF

    """
    width = 2 * psf_width * ref_wl_ind / (y - y0)
    return width
 
def calc_scaled_psf_row(psf_row, center_row, scale_factors):
    """
    For a PSf centered on some row, calculate where it goes after being scaled
    """
    y = (psf_row-center_row)*scale_factors + center_row
    return y

def invert_scaled_psf_row(
    scaled_rows : float | np.ndarray,
    psf_row : int | float,
	center_row : int | float,
	ref_wl : float,
	wl_pixscale : float,
	wl0,
) -> float | np.ndarray[float]:
    """
    Find the column at which a PSF located at `psf_row` in the original image
    crossed the given rows in the scaled image.

    Parameters
    ----------
    scaled_rows : float | np.ndarray
      the rows in scaled space where you want to find the crossing column
    psf_row : float
      the row in the original image with the source
    center_row : float
      the center of the scaling
    ref_wl : float
      the reference wavelength for the scaling, converted to Angstrom
	wl_pixscale : float
      the pixel scale. CD1_1 in the SCI header.
    wl0 : the wavelength of the 0th column

    Output
    ------
    cols : float | np.ndarray
      the columns at which the PSF crosses the rows
    """
    scaled_sep = scaled_rows - center_row
    orig_sep = psf_row - center_row
    cols = (wl_pixscale**-1) * ( ref_wl*(orig_sep/scaled_sep) - wl0 )
    return cols

def calc_wl_mask_position(
    y0 : float,
    y1 : int,
	ycen : float,
	psf_width : float,
    wlsol : np.ndarray,
	ref_wl_ind : float,
    wl_pixscale : float,
) -> tuple[float, float]:
    """
    Compute the center and width of the mask, along the wavelength axis, required to
    mask out a rescaled companion PSF.
    Returns a tuple of [mask_center, mask_width].
    Note that the mask width does NOT depend on y0, only on y1, so you can use
    it regardless of the value of y0.

    Parameters
    ----------
    y0 : float
      the location of the source in the original (unscaled) image
    y1 : int
      the row you are interpolating
	ycen: float
      the central row from which the scaling is computed
	ref_wl_ind : float
      the index of the reference wavelength for the scaling
	psf_width : float
      the spatial (y) half-size of the psf

    Output
    ------
    mask_width : float
      the mask-width in wavelength space required to mask out the spatially-rescaled PSF

    """
    width = 2 * psf_width * ref_wl_ind / (y1 - ycen)
    # center = ref_wl_ind * (y0-ycen) / (y1-ycen)
    ref_wl = wlsol[ref_wl_ind]
    wl0 = wlsol[0]
    center = invert_scaled_psf_row(y1, y0, ycen, ref_wl, wl_pixscale, wl0)
    return center, width


def descale_signal(
        residual_img, ytest, ycen, obs, ref_wl_ind
) -> np.ndarray:
    """
    From a wavelength-scaled residual image, use a simple algorithm to estimate
    the signal in unscaled space:

    - Compute the position of the signal in row, col coordinates.
    - For each column, take the two closest rows and compute their distance-weighted mean
    - return one value for each column

    Parameters
    ----------
    define your parameters

    Output
    ------
    Define your output

    """
    cols = np.arange(residual_img.shape[1])
    scale_factors = obs.wlsol[ref_wl_ind] / obs.wlsol
    signal_rows = calc_scaled_psf_row(ytest, ycen, scale_factors)
    signal = np.zeros_like(cols)*np.nan
    for c in cols:
        r = signal_rows[c]
        r_lo, r_hi = [f(r).astype(int) for f in (np.floor, np.ceil)]
        weights = np.abs(r-r_lo)**-2, np.abs(r_hi-r)**-2
        signal[c] = residual_img[[r_lo,r_hi], c]*weights / np.sum(weights)
    return signal



