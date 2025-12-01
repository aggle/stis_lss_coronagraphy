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



def calc_wl_mask_position(
    y0 : float,
    y1 : int,
	ycen : float,
	ref_wl_ind : float,
	psf_width : float
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
    center = ref_wl_ind * (y0-ycen) / (y1-ycen)
    return center, width

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
 
def interpolate_by_row(
        img : np.ndarray,
        mask_width : int = 3,
) -> np.ndarray :
    """
    Interpolate an image row-by-row. For each row, mask a region off and then predict it by interpolation.

    Parameters
    ----------
    img : np.ndarray
      2-D wavelength-scaled spectral image (speckles move across rows)
    mask_width : int = 3
      how many pixels before and after the test pixel to mask

    Output
    ------
    interp_img : np.ndarray
      the image as replaced by interpolated values
    """
    interp_img = np.empty_like(img)

    for col in np.arange(mask_width, img.shape[1]-mask_width):
        new_mask = np.zeros_like(img, dtype=bool)
        lb, ub = col-mask_width, col+mask_width+1
        lb = max([ 0, lb ])
        ub = min([ img.shape[1], ub ])
        new_mask[:, lb : ub] = True
        masked_img = np.stack([row[~row_mask] for row, row_mask in zip(img, new_mask)])
        # interp along the column axis
        interp_row = interpolate.CubicSpline(
            np.arange(img.shape[1])[~new_mask[0]],
            masked_img, 
            axis=1
        )
        interp_img[:, col] = interp_row(col)
    return interp_img
