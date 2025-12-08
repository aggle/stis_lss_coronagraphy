import numpy as np
from scipy import ndimage
from scipy import interpolate

from astropy import units
from astropy.wcs import WCS

from coronspec_tools import observing_sequence

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
      how much to scale each position. essentially, the wavelength solution
      normalized to some index.
      defined with the convention wlsol[ref_wl]/wlsol

    Output
    ------
    scaled_img : np.ndarray
      the image rescale to some wavelength index
    """
    row_coords, col_coords = np.mgrid[:img.shape[0], :img.shape[1]]
    row_sep = row_coords - center_row
    new_rows = (row_sep / scale_factors) + center_row
    scaled_img = ndimage.map_coordinates(
        img.copy(), [new_rows, col_coords], mode='nearest'
    )
    return scaled_img

def rescale_img_with_wcs(
    img : np.ndarray,
    wcs : WCS,
    star_sep : units.Quantity,
    scale_factors : np.ndarray
) -> np.ndarray:
    """
    compute the rescaling factors for each wavelength slice

    Parameters
    ----------
    img : np.ndarray
      2-D spectral image. Rows are separation, Cols are wavelength
    wcs : astropy.wcs.WCS
      the wcs corresponding to the image
    star_sep : units.Quantity[deg]
      the position of the target along the spatial axis
    scale_factors : np.ndarray
      how much to scale each position. essentially, the wavelength solution
      normalized to some index.
      defined using the convention wlsol[ref_wl]/wlsol

    Output
    ------
    scaled_img : np.ndarray
      the image rescale to some wavelength index
    """
    # convert the pixels to wavelength and separation
    rows, cols = np.mgrid[:img.shape[0], :img.shape[1]]
    wls, seps = wcs.pixel_to_world(cols, rows)
    seps -= star_sep
    # scale the separations and convert the scaled separations back to pixels
    scaled_seps = seps / scale_factors
    scaled_cols, scaled_rows = wcs.world_to_pixel(wls, scaled_seps)

    scaled_img = ndimage.map_coordinates(
        img.copy(), [scaled_rows, scaled_cols], mode='nearest'
    )
    return scaled_img

def compute_scaled_psf_trace_wcs(
    sep : units.Quantity,
    wcs : WCS,
    wlsol : units.Quantity,
    ref_wl_ind : int,
):
    """
    For an off-axis PSF at separation `sep` in deg, get the scaled row, col values
    """
    scale = wlsol[ref_wl_ind]/wlsol
    sep = sep * scale
    row, col = wcs.world_to_pixel(wlsol.to(units.m), sep.to(units.deg))
    return row, col


def compute_scaled_psf_trace(psf_row, center_row, scale_factors):
    """
    For a PSf centered on some row, calculate where it goes after being scaled
    one scale factor for each column
    scale_factor := wlsol/wlsol[ref_wl]
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

def compute_mask_halfwidth(
    y1 : int,
	ycen : float,
	psf_halfwidth : float,
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
    y1 : int
      the row you are interpolating
	ycen: float
      the central row from which the scaling is computed
	psf_halfwidth : float
      the spatial (y) half-size of the psf
    wlsol : np.ndarray
      unitless wavelength solution, in Angstrom
	ref_wl_ind : float
      the index of the reference wavelength for the scaling
    wl_pixscale : float
      how many wavelengths per pixel in the row axis

    Output
    ------
    half_width : float
      the half-width in columns required to mask out the spatially-rescaled PSF

    """
    half_width = wlsol.min()/wl_pixscale * psf_halfwidth / np.abs(y1-ycen)
    return half_width

def calc_wl_mask_position(
    y0 : float,
    y1 : int,
	ycen : float,
	psf_halfwidth : float,
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
	psf_halfwidth : float
      the spatial (y) half-size of the psf
    wlsol : np.ndarray
      unitless wavelength solution, in Angstrom
	ref_wl_ind : float
      the index of the reference wavelength for the scaling
    wl_pixscale : float
      how many wavelengths per pixel in the row axis

    Output
    ------
    mask_width : float
      the mask-width in wavelength space required to mask out the spatially-rescaled PSF

    """
    if ref_wl_ind < 0:
        ref_wl_ind = np.arange(wlsol.size)[ref_wl_ind]
    # compute the center
    ref_wl = wlsol[ref_wl_ind]
    wl0 = wlsol[0]
    center = invert_scaled_psf_row(y1, y0, ycen, ref_wl, wl_pixscale, wl0)
    # compute the width
    width = compute_mask_halfwidth(y1, ycen, psf_halfwidth, wlsol, ref_wl_ind, wl_pixscale)
    return center, 2*width


def descale_signal(
    residual_img : np.ndarray,
	ytest : float,
	ycen : float,
    wl_scaling : np.ndarray,
) -> np.ndarray:
    """
    From a wavelength-scaled residual image, use a simple algorithm to project the signal from an off-axis PSF back
    into unscaled space:

    - Compute the position of the signal in row, col coordinates.
    - For each column, take the two closest rows and compute their distance-weighted mean
    - return one value for each column

    Parameters
    ----------
    residual_image : np.ndarray
      the psf-subtracted residual in scaled space
	ytest : float
      the unscaled row you are testing for the presence of a PSF
	ycen : float
      the reference row position for scaling
    wl_scaling : np.ndarray
      wlsol[ref_wl]/wlsol

    Output
    ------
    signal : np.ndarray
      1-d row corresponding to the inferred signal in the unscaled row

    """
    cols = np.arange(residual_img.shape[1])
    signal_rows = compute_scaled_psf_trace(ytest, ycen, wl_scaling)
    signal = np.zeros_like(cols)*np.nan
    for c in cols:
        r = signal_rows[c]
        r_lo, r_hi = [f(r).astype(int) for f in (np.floor, np.ceil)]
        weights = np.abs(r-r_lo)**-2, np.abs(r_hi-r)**-2
        signal[c] = np.sum(residual_img[[r_lo,r_hi], c]*weights) / np.sum(weights)
    return signal

def construct_psf_model(
    scaled_img : np.ndarray,
    obs : observing_sequence.ObsSeq,
    y_test : int,
    y_ref : float,
    wl_ref_ind : int,
    psf_width : float = 5.
) -> np.ndarray :
    """
    Perform PSF interpolation and subtraction for a hypothetical source located
    at y_test, and return the residual.

    Parameters
    ----------
    scaled_img : np.ndarray
      the wavelength-scaled image
    obs : observing_sequence.ObsSeq
      the ObsSeq object carrying the observation-related information
    y_test : float
      the position of a hypothetical source, in pixels, along the spatial axis of the provided image
    y_ref : float
      the reference position for the wavelength scaling
    wl_ref_ind : int
      the reference wavelength index for wavelength scaling
    psf_width : float = 5.
      the full width of the PSF along the spatial axis, used for masking

    Output
    ------
    residual : np.ndarray
      the result of scaled_img - psf_model

    """
    col_inds = np.arange(scaled_img.shape[1])
    psf_model = scaled_img.copy()
    scale_factors = obs.wlsol[wl_ref_ind]/obs.wlsol
    scaled_rows = compute_scaled_psf_trace(y_test, y_ref, scale_factors)
    unique_rows = np.arange(np.floor(scaled_rows.min()), np.ceil(scaled_rows.max()), dtype=int)
    for row_ind in unique_rows:
        # compute the center and width of a scaled PSF projected across a row
        mask_center, mask_width = calc_wl_mask_position(
            y_test,
            row_ind, 
            y_ref, 
            psf_width, 
            obs.wlsol.to(units.Angstrom).value, 
            wl_ref_ind, 
            obs.hdrs['occ']['sci']['CD1_1']
        )
        mask_range = np.round([mask_center-mask_width/2, mask_center+mask_width/2]).astype(int)
        mask = np.zeros(scaled_img.shape[1]).astype(bool)
        mask[mask_range[0]:mask_range[1]] = True 
        masked_row = np.ma.masked_array(scaled_img[row_ind], mask=mask)
        interp_row = interpolate.Akima1DInterpolator(
                col_inds[~masked_row.mask],
                masked_row[~masked_row.mask], 
            )(col_inds)
        psf_model[row_ind] = interp_row
    return psf_model


def model_and_subtract_target(
    scaled_img : np.ndarray,
    obs : observing_sequence.ObsSeq,
    y_test : int,
    y_ref : float,
    wl_ref_ind : int,
    psf_width : float = 5.
) -> np.ndarray :
    """
    Perform PSF interpolation and subtraction for a hypothetical source located
    at y_test, and return the residual.

    Parameters
    ----------
    scaled_img : np.ndarray
      the wavelength-scaled image
    obs : observing_sequence.ObsSeq
      the ObsSeq object carrying the observation-related information
    y_test : float
      the position of a hypothetical source, in pixels, along the spatial axis of the provided image
    y_ref : float
      the reference position for the wavelength scaling
    wl_ref_ind : int
      the reference wavelength index for wavelength scaling
    psf_width : float = 5.
      the full width of the PSF along the spatial axis, used for masking

    Output
    ------
    residual : np.ndarray
      the result of scaled_img - psf_model

    """
    col_inds = np.arange(scaled_img.shape[1])
    psf_model = scaled_img.copy()
    scale_factors = obs.wlsol/obs.wlsol[wl_ref_ind]
    scaled_rows = compute_scaled_psf_trace(y_test, y_ref, scale_factors)
    unique_rows = np.arange(
        np.floor(scaled_rows.min()), np.ceil(scaled_rows.max()),
        dtype=int
    )
    masks = {}
    for row_ind in unique_rows:
        # compute the center and width of a scaled PSF projected across a row
        mask_center, mask_width = calc_wl_mask_position(
            y_test,
            row_ind,
            y_ref,
            psf_width,
            obs.wlsol.to(units.Angstrom).value,
            wl_ref_ind,
            obs.hdrs['occ']['sci']['CD1_1']
        )
        mask = make_row_mask(obs.wlsol.size, mask_center, mask_width)
        masks[row_ind] = mask
        if mask.all():
            psf_model[row_ind] = scaled_img[row_ind][:]
        interp_row = fit_under_psf(col_inds, scaled_img[row_ind], mask)
        psf_model[row_ind] = interp_row

    residual = scaled_img - psf_model
    return unique_rows, masks, psf_model, residual

def make_row_mask(npix, center, width):
    """
    Parameters
    ----------
    npix : int
      the number of pixels in the row
    center : float
      the center of the mask, in pixels
    width : float
      the full width of the mask, in pixels

    Output
    ------
    mask : np.ndarray[bool]
      a boolean array that is True *inside* the mask region and False elsewhere
    """
    if center < 0 or center > npix:
        mask = np.ones(npix).astype(bool)
    else:
        mask_range = np.round(
            [center-width/2, center+width/2]
        ).astype(int)
        mask = np.zeros(npix).astype(bool)
        mask[mask_range[0]:mask_range[1]] = True
    return mask

def mask_range_to_bool(mask_range : tuple[float, float], row_size : int):
    """
    Convert a mask range into a boolean array
    mask_range : tuple[float]
      tuple of (lower bound, upper bound)
    Output
    ------
    mask : np.array
      a boolean array of row_size
    """
    # make a mask that is everywhere False
    mask = np.zeros(row_size, dtype=bool)
    # now fill the masked region with True
    mask_lb, mask_ub = mask_range 
    mask_lb = np.floor(max([0, mask_lb])).astype(int)
    mask_ub = np.ceil(min([mask_ub, row_size])).astype(int)
    mask[mask_lb:mask_ub] = True
    return mask

def fit_under_psf(
        col_inds : np.ndarray,
        data : np.ndarray,
        mask : np.ndarray,
) -> np.ndarray :
    """
    Infer the values of the data array under the test PSF.

    Parameters
    ----------
    col : np.ndarray[int]
      the column indices
    data : np.ndarray[float]
      the row of data
    mask : np.ndarray[bool]
      a mask that is true under the test psf

    Output
    ------
    bgnd : np.ndarray
      the data array with the masked values filled

    """
    # 1-D interpolation
    masked_row = np.ma.masked_array(data, mask=mask)
    psf_model_func = interpolate.Akima1DInterpolator(
        col_inds[~mask],
        data[~mask],
    )
    psf_model = data.copy()
    psf_model[masked_row.mask] = psf_model_func[np.where(masked_row.mask)[0]]

    # # fit a 2-D polynomial
    # mask_lb, mask_ub = np.where(mask)[0][[ 0, -1 ]] + np.array([0, 1])
    # fit_lb = max([mask_lb-30, 0]), mask_lb
    # fit_ub = mask_ub+1, min([mask_ub+1+30, data.size])
    # fit_cols = np.concatenate([
    #     np.arange(fit_lb[0], fit_lb[1]),
    #     np.arange(fit_ub[0], fit_ub[1]),
    # ])
    # fit_data = data[fit_cols]
    # # 2nd-order polynomial fit
    # if len(fit_data) < 1:
    #     print("No data to fit!")
    #     return data
    # poly2 = np.polynomial.Polynomial.fit(fit_cols, fit_data, 1)
    # psf_model = data.copy()
    # psf_model[mask] = poly2(np.arange(mask_lb, mask_ub))
    return psf_model


class SDI:
    def __init__(self, obs: observing_sequence.ObsSeq, psf_halfwidth=5):
        """
        A class that helps with SDI operations. Using a class is helpful
        because you can track information like the reference wavelength
        """
        self.obs = obs
        self.wl_pixscale = obs.hdrs['occ']['sci']['CD1_1']
        self.ref_wl_ind = len(obs.wlsol)-1
        self.scale_factors = self.obs.wlsol[self.ref_wl_ind]/self.obs.wlsol
        self.scaled_stamp = np.zeros_like(obs.occ_stamp.data)
        self.psf_halfwidth = psf_halfwidth

    def compute_scaled_stamp(self, ref_wl_ind, stamp, stamp_center):
        scale_factors = self.obs.wlsol[ref_wl_ind]/self.obs.wlsol
        scaled_stamp = rescale_img(
            stamp,
            stamp_center,
            scale_factors
        )
        self.scaled_stamp = scaled_stamp
        self.ref_wl_ind = ref_wl_ind
        self.scale_factors = scale_factors

    def subtract_target_model(self, target_row_ind, psf_halfwidth = None):
        if psf_halfwidth is None:
            psf_halfwidth = self.psf_halfwidth
        y = compute_scaled_psf_trace(
            target_row_ind, 
            self.obs.occ_stamp_center,
            self.scale_factors
        )
        trace_rows = np.arange(np.floor(y.min()), np.ceil(y.max())+1, dtype=int)
        scaled_region = self.scaled_stamp[trace_rows]
        psf_model = self.model_target_row(target_row_ind, psf_halfwidth)
        return scaled_region - psf_model

    def descale_residual_flux(self, residual_img, y_vals):
        """
        y_vals : np.ndarray
          the y value in each column for which to estimate the signal
        """
        cols = np.arange(residual_img.shape[1])
        signal = np.zeros_like(cols)*np.nan
        for c in cols:
            r = y_vals[c]
            r_lo, r_hi = [f(r).astype(int) for f in (np.floor, np.ceil)]
            weights = np.abs(r-r_lo)**-2, np.abs(r_hi-r)**-2
            signal[c] = np.sum(residual_img[[r_lo,r_hi], c]*weights) / np.sum(weights)
        return signal

    def model_target_row(self, target_row_ind, psf_halfwidth = None):
        """
        Generate a model for the given row of an unscaled stamp

        Parameters
        ----------
        target_row_ind : float
          the position of the *unscaled* stamp at which you wish to model the primary PSF

        Output
        ------
        psf_model : np.ndarray
          the model of the speckle field, in scaled space
        """
        if psf_halfwidth is None:
            psf_halfwidth = self.psf_halfwidth
        x = np.arange(self.scaled_stamp.shape[1])
        y = compute_scaled_psf_trace(
            target_row_ind, 
            self.obs.occ_stamp_center,
            self.scale_factors
        )
        trace_rows = np.arange(np.floor(y.min()), np.ceil(y.max())+1, dtype=int)
        psf_model = np.zeros((trace_rows.size, x.size))
        for i, scaled_row_ind in enumerate(trace_rows):
            model_row = self.model_scaled_row(target_row_ind, scaled_row_ind, psf_halfwidth)
            psf_model[i] = model_row
        return psf_model


    def model_scaled_row(self, target_row_ind : int, scaled_row_ind : int, psf_halfwidth = None):
        """Model the PSF under the hypothetical companion from a target row, at a single scaled row"""
        if psf_halfwidth is None:
            psf_halfwidth = self.psf_halfwidth
        # compute the mask
        mask_range = self.compute_row_mask(target_row_ind, scaled_row_ind, psf_halfwidth)
        mask = mask_range_to_bool(mask_range, self.obs.wlsol.size)
        # model the row
        scaled_row = self.scaled_stamp[scaled_row_ind]
        masked_row = np.ma.masked_array(scaled_row, mask)
        psf_model = self.fit_masked_data(masked_row)
        return psf_model

    def compute_row_mask(self, target_row, scaled_row, psf_halfwidth=5):
        """
        For a given target row and scaled row, return lower and upper bounds of
        the masked region in columns
        """
        wlsol = self.obs.wlsol.to("Angstrom").value
        center = invert_scaled_psf_row(
            scaled_row,
            target_row,
            self.obs.occ_stamp_center,
            wlsol[self.ref_wl_ind],
            self.wl_pixscale,
            wlsol.min()
        )
        mask_halfwidth = compute_mask_halfwidth(
            y1 = scaled_row,
            ycen = self.obs.occ_stamp_center,
            psf_halfwidth = psf_halfwidth,
            wlsol = wlsol,
            ref_wl_ind = self.ref_wl_ind,
            wl_pixscale = self.wl_pixscale,
        )/2
        mask_lb, mask_ub = center - mask_halfwidth, center + mask_halfwidth
        return mask_lb, mask_ub

    def fit_masked_data(self, masked_row):
        """
        Replace the masked data in the row with some function
        """
        psf_model = masked_row.data.copy()
        col_inds = np.arange(masked_row.size)
        mask = masked_row.mask
        if col_inds[~mask].size == 0:
            # everything is maskde
            psf_model = masked_row.data * np.nan
        elif col_inds[mask].size == col_inds.size:
            # nothing is masked
            psf_model = masked_row.data
        else:
            # psf_model_func = interpolate.Akima1DInterpolator(
            #     col_inds[~mask],
            #     masked_row[~mask],
            # )
            lb, ub = np.where(mask)[0][[0, -1]]
            lb_range = [max([lb - 20, 0]), lb]
            ub_range = [ub, min([col_inds.size, ub+20])]
            fit_pix = np.concatenate([
                col_inds[lb_range[0]:lb_range[1]],
                col_inds[ub_range[0]:ub_range[1]]
            ])
            psf_model_func = np.polynomial.Polynomial.fit(fit_pix, masked_row[fit_pix], 1)

            psf_model[mask] = psf_model_func(np.where(masked_row.mask)[0])
        return psf_model
