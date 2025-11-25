"""
Useful tools
"""
from pathlib import Path

import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.stats import SigmaClip

from photutils.centroids import centroid_2dg as centroid_func
from scipy import ndimage

img_vrange = lambda img, vmin=0.01, vmax=0.99: dict(zip(['vmin','vmax'], np.nanquantile(img, [vmin, vmax])))

def get_stamp_shape(stamp : int | np.ndarray | pd.Series) -> np.ndarray:
    """
    Get the 2-d shape of a stamp given an int, array, or series

    Parameters
    ----------
    stamp : int | np.ndarray | pd.Series
      Either the length of one side, a 2-D image, or a Series of 2-D images
    Output
    ------
    shape : tuple[int]
      the (row, col) shape
    """
    if isinstance(stamp, int):
        shape = np.tile(stamp, 2)
    elif isinstance(stamp, pd.Series):
        shape = np.stack(stamp.values).shape[-2:]
    else:
        shape = stamp.shape[-2:]
    return shape

def get_stamp_center(stamp : int | np.ndarray | pd.Series) -> np.ndarray:
    """
    Get the central pixel of a stamp or cube

    Parameters
    ----------
    stamp : int | np.ndarray | pd.Series
      Either the length of one side, a 2-D image, or a Series of 2-D images
    Output
    ------
    center : np.ndarray
      the center in (x, y)/(col, row) format
    """
    shape = get_stamp_shape(stamp)
    center = np.floor(np.array(shape)/2).astype(int)
    return center

def compute_psf_center(stamp, pad=2):
    """
    Compute the PSF center in a 3x3 box around the nominal center
    return center in (x, y) convention
    """
    center = get_stamp_center(stamp)
    ll = center - pad # lower left corner
    rows, cols = (center[1]-pad, center[1]+pad+1), (center[0]-pad, center[1]+pad+1)
    fit_stamp = stamp[rows[0]:rows[1], cols[0]:cols[1]]
    psf_center = centroid_func(fit_stamp) + ll
    return psf_center

def shift_stamp_to_center(stamp, pad=3):
    center = get_stamp_center(stamp)
    # assume the center is already in the correct pixel and we want only a
    # subpixel shift cut out a small region around the center to avoid affects
    # from possible nearby companions
    psf_center = compute_psf_center(stamp)
    shift = -(psf_center-center)[::-1]
    shifted_img = ndimage.shift(stamp, shift, mode='reflect')
    return shifted_img


def scale_stamp(stamp):
    return (stamp - np.nanmin(stamp))/np.ptp(stamp)


def row_get_psf_stamp_position(row, stamp_size=0):
    """Use the catalog position to get the PSF center in the stamp"""
    xy = np.array(row[['x', 'y']] % 1) - 0.5
    if stamp_size != 0:
        stamp_center = get_stamp_center(stamp_size)
        xy += stamp_center
    return xy

def get_pix_separation_from_center(stamp_size):
    """Get a map of the separation of each pixel from the center"""
    center = get_stamp_center(stamp_size)
    grid = (np.mgrid[:stamp_size, :stamp_size] - center[:, None, None])
    sep_map = np.linalg.norm(grid, axis=0)
    return sep_map

def center_to_ll_coords(stamp_size, pix):
    """
    Convert center-origin coordinates to ll-origin coordinates
    returns pix in (x, y) convention
    """
    center = get_stamp_center(stamp_size)
    ll_coord = center + np.array(pix)
    return ll_coord

def ll_to_center_coords(stamp_size, pix):
    """Convert center-origin coordinates to ll-origin coordinates"""
    center = get_stamp_center(stamp_size)
    center_coord = np.array(pix) - center
    return center_coord

def get_annuli(
        stamp_size : int,
        width : int = 2,
        rolling : bool = False,
) -> list:
    """
    Return a list of indices that define successive annuli of the stamp

    Parameters
    ----------
    stamp_size : int
      The one-sided size of the stamp
    width : int = 2
      width in pixels of the annuli
    rolling : bool = False
      if True, the annuli overlap by half the width

    Output
    ------
    annuli : dict
      Key is the whole-integer radius. Value is the list of coordinates

    """
    radii = get_pix_separation_from_center(stamp_size)
    overlap = width/2 if rolling else 0
    max_width = int(np.floor(stamp_size/2)) - width
    annuli = []
    for i in np.arange(0, max_width+overlap+1, width - overlap):
         pix = np.where((radii >= i) & (radii < i+width))
         sep = np.mean(radii[pix])
         annuli.append((sep, pix))
    return annuli


def find_outliers(
        img : np.ndarray[float],
        width : int = 5,
        thresh : float = 5.
) -> np.ndarray[bool] :
    """
    Identify outliers in an image. Search in a box of size `window` for pixels
    that are a std of `thresh` away from the rest of the population. Return a
    boolean mask where True is an outlier and False is not.

    Parameters
    ----------
    img : np.ndarray[float]
      2-D image data
    window : int = 5
      the 1-D box size for the test population
    thresh : float = 5
      sigma threshold for classifying a pixel as an outlier

    Output
    ------
    mask : np.ndarray[bool]
      An array with the same shape as img where True is an outlier pixel and
      False is not.

    """
    half_width = int(np.round((width - 1)/2))
    shape = img.shape
    grid = np.stack(
        [np.ravel(axis) for axis in np.mgrid[:shape[0],:shape[1]]],
        axis=1
    )
    def get_window_pix(r, c, shape):
        rlo = max(r-half_width, 0)
        rhi = min(r+half_width, shape[0])
        clo = max(c-half_width, 0)
        chi = min(c+half_width, shape[1])
        return np.mgrid[rlo:rhi,clo:chi]

    clipper = SigmaClip(sigma=thresh)
    mask = np.zeros_like(img)
    for r, c in grid:
        window_pix = get_window_pix(r, c, shape)
        stats = clipper(img[*window_pix])
        # set the mask value for the pixel r, c
        val = stats[(window_pix[0] == r) & (window_pix[1] == c)].mask.squeeze()
        mask[r, c] = val
    return mask

def get_unocc_spec_stamp(
        unocc_file : str | Path,
        size : int = 10,
) -> np.ndarray :
    """
    Extract a stamp of the unocculted spectrum from the unocc file

    Parameters
    ----------
    unocc_file : str | Path
      readable path to the file with the unocculted spectrum
    size : int = 10
      the upper and lower buffer around the center of the PSF (full height = 2*size + 1)
    Output
    ------
    unocc_stamp : np.ndarray
      2-D array with the wavelength axis along the columns

    """
    unocc_img = fits.getdata(str(unocc_file), 'SCI')
    unocc_img_row = np.nanargmax(np.nansum(unocc_img, axis=1))
    unocc_stamp = unocc_img[unocc_img_row-size:unocc_img_row+size+1, :]
    return unocc_stamp

def inject_template(
    target_img : np.ndarray,
    template : np.ndarray,
    row : int,
    scale : float = 1.,
) -> np.ndarray :
    """
    Inject a template spectrum into the target image

    Parameters
    ----------
    target_img : np.ndarray
      the image to add the spectrum into. Has dimensions N rows x M cols
    template : np.ndarray
      the template stamp that will be added to the image. Has dimensions T < N rows, M cols
    row : int
      the central row of the injection site
    scale : float = 1.
      multiply the template by this factor
    Output
    ------
    inj_img : np.ndarray
      N x M image with the injected template

    """

    # pad the template to the same dimensions as the target
    height, width = template.shape
    dheight = int((height - height%2)/2)
    pad_below = row - dheight
    pad_above = target_img.shape[0]-(row + dheight)-1
    template = np.pad(template, [(pad_below, pad_above), (0, 0)])
    inj_img = target_img + ( template * scale )
    return inj_img


# some useful math tools
def rotate_point(xy, theta, center = 0):
    """
    rotate point xy by angle around some center
    xy : tuple[float]
      the (x, y) coordinates of the point to rotate
    theta : float
      the angle from the x-axis in degrees
    center : tuple[float] or 0
      the center about which to rotate
    """
    xy = np.array(xy)
    center = np.array(center)
    theta = np.deg2rad(theta)
    matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    xy_rot = np.inner(matrix, xy-center).T + center
    return xy_rot

def make_radial_mask(
        shape : tuple[int, int],
        start_pos : tuple[int, int],
        theta : float,
        width : int,
) -> np.ndarray[bool] :
    """
    Make a boolean mask that masks a ray at angle theta starting from position start_pos, with width `width`

    Parameters
    ----------
    shape : tuple[int]
      the (row, col) shape of the final mask
    start_pos : tuple[int]
      the (row, col) position to start from
    theta : float
      the angle of the ray, measured from the x-axis in degrees
    width : int
      the width of the mask, in pixels

    Output
    ------
    mask : np.ndarary[bool]
      False inside the ray, True outside the ray
    """
    mask = np.ones(shape, dtype=bool)
    y0, x0 = start_pos
    slope = np.tan(np.deg2rad(theta))
    x = np.arange(x0, shape[1], dtype=int)
    line = lambda x, yc: (slope * (x - x0) + yc).astype(int)
    ub = line(x, y0 + width/2)
    lb = line(x, y0 - width/2)
    for i, j, k in zip(x, lb, ub):
        mask[j:k+1, i] = False
    return mask


def make_line_func(x0, y0, theta):
    slope = np.tan(np.deg2rad(theta))
    line = lambda x: slope * (x - x0) + y0
    return line
