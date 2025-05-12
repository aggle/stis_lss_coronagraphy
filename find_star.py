"""
Tools for finding the position of the star in the occulted (or unocculted!) data
"""

import numpy as np

from radonCenter import radonCenter, radonOffset


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
