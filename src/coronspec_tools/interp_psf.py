"""
Interpolate the PSF and subtract it
"""
import itertools
 
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
        lb = max(col, lb)
        ub = min(img.shape[1], ub)
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

def interpolate_by_row_2d(
        img : np.ndarray,
        row_mask : int = 3,
        col_mask : int = 3,
) -> np.ndarray :
    """
    Interpolate an image row-by-row. For each row, mask a region off and then predict it by interpolation.

    Parameters
    ----------
    img : np.ndarray
      2-D wavelength-scaled spectral image (speckles move across rows)
    row_mask : int = 3
      how many pixels before and after the test pixel to mask

    Output
    ------
    interp_img : np.ndarray
      the image as replaced by interpolated values
    """
    interp_img = np.empty_like(img)

    for col in np.arange(row_mask, img.shape[1]):
        new_mask = np.zeros_like(img, dtype=bool)
        lb, ub = col-row_mask, col+row_mask+1
        lb = max(col, lb)
        ub = min(img.shape[1], ub)
        new_mask[:, lb : ub] = True
        masked_img = np.stack([row[~row_mask] for row, row_mask in zip(img, new_mask)])
        # interp along the column axis
        cols = np.arange(img.shape[1])[~new_mask[0]],
        interp_row = interpolate.CubicSpline(
            np.arange(img.shape[1])[~new_mask[0]],
            masked_img, 
            axis=1
        )
        interp_img[:, col] = interp_row(col)
    return interp_img

def interpolate_slice(
        img : np.ndarray,
        row : int, col : int,
        drow : int, dcol : int,
):
    """
    Write a 2-D interpolation for a slice taken out of an image
    """
    row_lb, row_ub = row - drow, row + drow + 1
    row_lb, row_ub = max([row_lb, 0]), min([row_ub, img.shape[0]])
    col_lb, col_ub = col - dcol, col + dcol + 1
    col_lb, col_ub = max([col_lb, 0]), min([col_ub, img.shape[1]])
    row_ind, col_ind = np.concatenate(
        [np.concatenate(
            [np.mgrid[0:row_lb+1, 0:col_lb+1], np.mgrid[row_ub:img.shape[0], 0:col_lb+1]], 
            axis=1
        ),
        np.concatenate(
            [np.mgrid[0:row_lb+1, col_ub:img.shape[1]], np.mgrid[row_ub:img.shape[0], col_ub:img.shape[1]]],
            axis=-2
        )],
        axis=-1
    )
    all_points = tuple(zip(*list(tuple(i.ravel()) for i in np.mgrid[:img.shape[0], :img.shape[1]])))
    mask_points = tuple(itertools.product(range(row_lb, row_ub), range(col_lb, col_ub)))
    unmasked_points = list(set(all_points).difference(mask_points))
    img_slice = img[row_ind, col_ind]
    # points = np.stack([ row_ind.ravel(), col_ind.ravel() ]).T
    interp_func = interpolate.CloughTocher2DInterpolator(
        unmasked_points, img_slice.ravel()
    )
    return interp_func(all_points).reshape(img.shape)

    
