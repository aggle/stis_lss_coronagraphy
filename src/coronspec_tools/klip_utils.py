import numpy as np
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

def align_and_scale(img, center_pix,  wlmap):
    pass
