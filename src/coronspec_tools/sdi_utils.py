import numpy as np
from scipy import ndimage
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

def align_and_scale(
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
    scaled_img = ndimage.map_coordinates(img.copy(), [new_rows, col_coords] )
    return scaled_img


 
