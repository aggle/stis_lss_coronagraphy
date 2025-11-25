"""
Interpolate the PSF and subtract it
"""

import numpy as np
from scipy import interpolate

from astropy import units

from coronspec_tools.observing_sequence import ObsSeq


def interpolate_column(
    img : np.ndarray,
    col_num : int,
    mask_width : int = 3,
) -> np.ndarray :
    """
    Interpolate the values for a column by masking it off and making an
    interpolation function from the rest of the image.

    Parameters
    ----------
    img : np.ndarray
      the image to interpolate
    col_num : int
      the column to replace
    mask_width : int = 3
      how much margin on either side of the column to mask out

    Output
    ------
    col : np.ndarray
       the interpolated values for the column

    """
    mask_lb = max([0, col_num-mask_width])
    mask_ub = min([col_num+mask_width+1, img.shape[1]])
    masked_img = np.ma.masked_array(img, mask=False)
    masked_img.mask[:, mask_lb:mask_ub] = True
    
    coords = np.mgrid[:img.shape[0],:img.shape[1]]
    ycoords, xcoords = [i.ravel() for i in coords]
    yxcoords = np.array([ycoords, xcoords]).T
    interp_func = interpolate.CloughTocher2DInterpolator(
        yxcoords[~masked_img.mask.ravel()],
        masked_img.flat[~masked_img.mask.ravel()]
    )
    img_interp = interp_func(yxcoords).reshape(img.shape)
    col = img_interp[:, col_num]
    return col

def interpolate_row(
    img : np.ndarray,
    row_num : int,
    mask_width : int = 3,
) -> np.ndarray :
    """
    Interpolate the values for a column by masking it off and making an
    interpolation function from the rest of the image.

    Parameters
    ----------
    img : np.ndarray
      the image to interpolate
    col_num : int
      the column to replace
    mask_width : int = 3
      how much margin on either side of the column to mask out

    Output
    ------
    col : np.ndarray
       the interpolated values for the column

    """
    mask_lb = max([0, row_num-mask_width])
    mask_ub = min([row_num+mask_width+1, img.shape[0]])
    masked_img = np.ma.masked_array(img, mask=False)
    masked_img.mask[mask_lb:mask_ub, :] = True
    
    coords = np.mgrid[:img.shape[0],:img.shape[1]]
    ycoords, xcoords = [i.ravel() for i in coords]
    yxcoords = np.array([ycoords, xcoords]).T
    interp_func = interpolate.CloughTocher2DInterpolator(
        yxcoords[~masked_img.mask.ravel()],
        masked_img.flat[~masked_img.mask.ravel()]
    )
    img_interp = interp_func(yxcoords).reshape(img.shape)
    row = img_interp[row_num]
    return row

def make_interpolated_psf_model(
    img : np.ndarray,
    mask_width : int = 3,
    row_or_col : str = 'row',
) -> np.ndarray:
    """
    Make an interpolated version of the PSF for subtraction.
    For each column, mask +- a few columns and interpolate over them.
    Construct an image from the interpolated version of each column.

    Parameters
    ----------
    define your parameters

    Output
    ------
    Define your output

    """
    interp_img = np.zeros_like(img)
    if row_or_col == 'col':
        for col_num in np.arange(img.shape[1]):
            interp_img[:, col_num] = interpolate_column(
                img, col_num, mask_width
            )
    else:
        for row_num in np.arange(img.shape[0]):
            interp_img[row_num] = interpolate_row(
                img, row_num, mask_width
            )
    return interp_img



def calc_source_movement(
    obs : ObsSeq,
    sep : units.Unit,
):
    """
    """
    pass
