import numpy as np
from scipy import ndimage
from scipy import interpolate

import pandas as pd

from astropy import units
from astropy.wcs import WCS

from coronspec_tools import observing_sequence

class SDI:
    def __init__(self, obs: observing_sequence.ObsSeq, ref_wl_ind = -1, psf_halfwidth=5):
        """
        A class that helps with SDI operations. Using a class is helpful
        because you can track information like the reference wavelength
        """
        self.initialize()
        self.obs = obs
        self.wl_pixscale = obs.hdrs['occ']['sci']['CD1_1']
        self.ref_wl_ind = ref_wl_ind
        self.scale_factors = self.obs.wlsol[self.ref_wl_ind]/self.obs.wlsol
        self.scaled_stamp = np.zeros_like(obs.occ_stamp.data)
        self.psf_halfwidth = psf_halfwidth

    def initialize(self):
        """Instantiate everything with a setter/getter structure"""
        self._ref_wl_ind = None


    @property
    def ref_wl_ind(self) -> int:
        return self._ref_wl_ind
    @ref_wl_ind.setter
    def ref_wl_ind(self, newval) -> None:
        # you may want to use this to trigger recomputing the scaled stamps
        if newval >= self.obs.wlsol.size:
            print(f"Error: Index too large ({newval} > {self.obs.wlsol.size-1})")
        if newval < 0:
            newval = len(self.obs.wlsol) + newval
        self._ref_wl_ind = newval
        return



    def compute_scaled_stamp(
            self,
            ref_wl_ind : int | None = None,
            stamp : np.ndarray | None = None,
            stamp_center : float | None = None,
    ) -> None:
        if ref_wl_ind is None:
            ref_wl_ind = self.ref_wl_ind
        # if stamp is None:
        stamp = self.obs.occ_stamp.data
        stamp_unc = self.obs.occ_stamp_unc.data
        # if stamp_center is None:
        stamp_center = self.obs.occ_stamp_center


        scale_factors = self.obs.wlsol[ref_wl_ind]/self.obs.wlsol
        self.ref_wl_ind = ref_wl_ind
        self.scale_factors = scale_factors

        scaled_stamp = rescale_img(
            stamp,
            stamp_center,
            scale_factors
        )
        scaled_stamp_unc = rescale_img(
            stamp_unc,
            stamp_center,
            scale_factors
        )
        self.scaled_stamp = scaled_stamp
        self.scaled_stamp_unc = scaled_stamp_unc
        return

    def descale_trace(self, scaled_img, y_vals, y_range):
        """
        Get the signal from the trace in a wavelength-scaled image
        
        y_vals : np.ndarray
          the y value in each column for which to estimate the signal
        y_range : np.ndarray
          the min and max rows covered by the scaled img
        """
        # valid = (y_vals > y_range[0]) & (y_vals < y_range[1]-1)
        # normalize the y-values to the image rows
        shifted_y_vals = y_vals - y_range[0]
        valid = (shifted_y_vals >= 0) & (shifted_y_vals < scaled_img.shape[0])
        cols = np.arange(scaled_img.shape[1])[valid]

        signal = np.zeros_like(shifted_y_vals)*np.nan
        for c in cols:
            r = shifted_y_vals[c]
            r_lo, r_hi = [f(r).astype(int) for f in (np.floor, np.ceil)]
            # make sure that r_lo and r_hi are in the bounds
            r_lo = min([max([0,r_lo ]), r_hi])
            r_hi = max([min(r_hi, scaled_img.shape[0]-1), r_lo])
            weights = np.abs(r-r_lo)**-2, np.abs(r_hi-r)**-2
            signal[c] = np.sum(scaled_img[[r_lo,r_hi], c]*weights) / np.sum(weights)
        return signal

    def model_target_row(
            self,
            target_row_ind,
            psf_halfwidth = None,
            model_kwargs = {},
    ) -> np.ma.masked_array:
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
        trace_rows = self.check_trace_rows(y)
        psf_model = np.ma.masked_array(np.zeros((trace_rows.size, x.size)))
        for i, scaled_row_ind in enumerate(trace_rows):
            model_row = self.model_scaled_row(target_row_ind, scaled_row_ind, psf_halfwidth, **model_kwargs)
            psf_model[i] = model_row
        return psf_model


    def check_trace_rows(self, trace_y):
        # make sure that the trace row indices are valid
        trace_rows = np.arange(np.floor(trace_y.min()), np.ceil(trace_y.max())+1, dtype=int)
        trace_rows = trace_rows[trace_rows >= 0]
        trace_rows = trace_rows[trace_rows < self.scaled_stamp.shape[0]]
        return trace_rows


    def model_scaled_row(
        self,
        target_row_ind : int,
        scaled_row_ind : int,
        psf_halfwidth = None,
        fit_pad : int = 200,
        fit_poly : float = 2,
    ) -> np.ma.masked_array:
        """
        Model the PSF under the hypothetical companion from a target row, at a single scaled row
        target_row_ind : int
          row in the occulted stamp to model
        scaled_row_ind : int
          row in the scaled stamp to model
        psf_halfwidth = None
          half-size of the PSF in rows (i.e. tune this to change how many columns you mask)
        fit_pad : int = 200
          how many columns on either side of the mask to use for fitting
        fit_poly : float = 2
          order of the polynomial to use for fitting

        """
        if scaled_row_ind > self.scaled_stamp.shape[0]-1:
            print("Scaled row index too large; returning nan")
            return np.zeros(self.scaled_stamp.shape[1])*np.nan
        if psf_halfwidth is None:
            psf_halfwidth = self.psf_halfwidth
        # compute the mask
        mask_range = self._compute_row_mask(target_row_ind, scaled_row_ind, psf_halfwidth)
        mask = mask_range_to_bool(mask_range, self.obs.wlsol.size)
        # model the row
        scaled_row = self.scaled_stamp[scaled_row_ind]
        masked_row = np.ma.masked_array(scaled_row, mask)
        masked_unc = np.ma.masked_array(self.scaled_stamp_unc[scaled_row_ind], mask=mask)
        psf_model = self._fit_masked_data(masked_row, fit_poly, fit_pad, masked_unc)
        psf_model = np.ma.masked_array(psf_model, mask=~masked_row.mask)
        return psf_model

    def _compute_row_mask(self, target_row, scaled_row, psf_halfwidth=None) -> tuple[float, float]:
        """
        For a given target row and scaled row, return lower and upper bounds of
        the masked region in columns

        Parameters
        ----------
        target_row : int
          row in the occulted stamp to model
        scaled_row : int
          row in the scaled stamp to model
        psf_halfwidth = None
          half-size of the PSF in rows (i.e. tune this to change how many columns you mask)

        Output
        ------
        mask_lb, mask_ub : tuple[float, float]
          a floating-point tuple of the mask lower and upper bounds

        """
        if psf_halfwidth is None:
            psf_halfwidth = self.psf_halfwidth
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
        )
        mask_lb, mask_ub = center - mask_halfwidth, center + mask_halfwidth
        return mask_lb, mask_ub

    def _fit_masked_data(
        self,
	    masked_row : np.ma.masked_array,
	    polynomial_order : int = 2,
	    pad  : int = 200,
        unc_array : np.ndarray | None = None,
    ):
        """
        Replace the masked data in the row with some function
	    masked_row : np.ma.masked_array
          array with the hypothetical signal region masked out
	    polynomial_order : int = 2
          order of the polynomial to use for fitting the masked region
	    pad : int = 200
          upper limit on how many pixels on either side of the mask to use
        unc_array : np.ndarray | None = None
          Uncertainties used to weight the fit, if given
        """
        psf_model = masked_row.data.copy()
        col_inds = np.arange(masked_row.size)
        mask = masked_row.mask
        if col_inds[~mask].size == 0:
            # everything is masked
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
            lb_range = [max([lb - pad, 0]), lb]
            ub_range = [ub, min([col_inds.size, ub+pad])]
            fit_pix = np.concatenate([
                col_inds[lb_range[0]:lb_range[1]],
                col_inds[ub_range[0]:ub_range[1]]
            ])
            if unc_array is not None:
                unc_array = 1/unc_array[fit_pix]**2
            psf_model_func = np.polynomial.Polynomial.fit(
                fit_pix,
                masked_row[fit_pix],
                polynomial_order,
                w = unc_array,
            )

            psf_model[mask] = psf_model_func(np.where(masked_row.mask)[0])
        return psf_model

    def _get_masks_for_target_row(
            self, target_row_ind,
    ) -> pd.Series:
        """
        Compute all the mask edges for a target row
        """
        trace = compute_scaled_psf_trace(
            target_row_ind,
            self.obs.occ_stamp_center,
            self.scale_factors
        )
        trace_rows = self.check_trace_rows(trace)
        masks = pd.Series({
            row: self._compute_row_mask(target_row_ind, row, self.psf_halfwidth)
            for row in trace_rows
        })
        return masks

    def generate_model_results_df(
            self, row_lo, row_hi, model_kwargs={},
    ) -> None:
        """
        Generate a dataframe where each row is a target row and has the
        following columns:

        'trace': row coordinates of the trace in each column
        'row_indices': the row indices covered by the trace
        'scaled_stamp': the section of the scaled stamp indexed by row_indices
        'model': the model of the scaled stamp
        'residual': scaled_stamp - model

        model_kwargs : these get passed along to model_scaled_row

        """
        # these are the rows we will investigate for signal
        target_row_indices = np.arange(row_lo, row_hi+1)
        # these are the row coordinates in scaled space for a hypothetical source in each of those rows
        target_row_traces = pd.Series({i: compute_scaled_psf_trace(
            i, 
            self.obs.occ_stamp_center,
            self.scale_factors
        ) for i in target_row_indices})
        # get the unique and valid row indices for each trace
        target_row_model_rows = target_row_traces.apply(self.check_trace_rows)
        # these are the scaled stamp regions covered by the trace
        target_row_stamps = pd.Series({
            i: self.scaled_stamp[target_row_model_rows[i]]
            for i in target_row_indices
        })
        target_row_stamp_uncs = pd.Series({
            i: self.scaled_stamp_unc[target_row_model_rows[i]]
            for i in target_row_indices
        })
        # these are the model stamps covering the trace region in the target rows
        target_row_models = pd.Series({
            i: self.model_target_row(i, model_kwargs=model_kwargs)
            for i in target_row_indices
        })
        # split the model data and masks
        target_row_masks = target_row_models.apply(lambda el: el.mask)
        target_row_models = target_row_models.apply(lambda el: el.data)
        # data - model
        target_row_residuals = target_row_stamps - target_row_models
        # put it all in one organized dataframe
        self.model_results = pd.concat({
            'trace': target_row_traces,
            'row_indices': target_row_model_rows,
            'scaled_stamp': target_row_stamps,
            'scaled_stamp_unc' : target_row_stamp_uncs,
            'stamp_mask': target_row_masks,
            'model': target_row_models,
            'residual': target_row_residuals,
        }, axis=1)
        return

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


# def model_and_subtract_target(
#     scaled_img : np.ndarray,
#     obs : observing_sequence.ObsSeq,
#     y_test : int,
#     y_ref : float,
#     wl_ref_ind : int,
#     psf_width : float = 5.
# ) -> np.ndarray :
#     """
#     Perform PSF interpolation and subtraction for a hypothetical source located
#     at y_test, and return the residual.

#     Parameters
#     ----------
#     scaled_img : np.ndarray
#       the wavelength-scaled image
#     obs : observing_sequence.ObsSeq
#       the ObsSeq object carrying the observation-related information
#     y_test : float
#       the position of a hypothetical source, in pixels, along the spatial axis of the provided image
#     y_ref : float
#       the reference position for the wavelength scaling
#     wl_ref_ind : int
#       the reference wavelength index for wavelength scaling
#     psf_width : float = 5.
#       the full width of the PSF along the spatial axis, used for masking

#     Output
#     ------
#     residual : np.ndarray
#       the result of scaled_img - psf_model

#     """
#     col_inds = np.arange(scaled_img.shape[1])
#     psf_model = scaled_img.copy()
#     scale_factors = obs.wlsol/obs.wlsol[wl_ref_ind]
#     scaled_rows = compute_scaled_psf_trace(y_test, y_ref, scale_factors)
#     unique_rows = np.arange(
#         np.floor(scaled_rows.min()), np.ceil(scaled_rows.max()),
#         dtype=int
#     )
#     masks = {}
#     for row_ind in unique_rows:
#         # compute the center and width of a scaled PSF projected across a row
#         mask_center, mask_width = calc_wl_mask_position(
#             y_test,
#             row_ind,
#             y_ref,
#             psf_width,
#             obs.wlsol.to(units.Angstrom).value,
#             wl_ref_ind,
#             obs.hdrs['occ']['sci']['CD1_1']
#         )
#         mask = make_row_mask(obs.wlsol.size, mask_center, mask_width)
#         masks[row_ind] = mask
#         if mask.all():
#             psf_model[row_ind] = scaled_img[row_ind][:]
#         interp_row = fit_under_psf(col_inds, scaled_img[row_ind], mask)
#         psf_model[row_ind] = interp_row

#     residual = scaled_img - psf_model
#     return unique_rows, masks, psf_model, residual

# def make_row_mask(npix, center, width):
#     """
#     Parameters
#     ----------
#     npix : int
#       the number of pixels in the row
#     center : float
#       the center of the mask, in pixels
#     width : float
#       the full width of the mask, in pixels

#     Output
#     ------
#     mask : np.ndarray[bool]
#       a boolean array that is True *inside* the mask region and False elsewhere
#     """
#     if center < 0 or center > npix:
#         mask = np.ones(npix).astype(bool)
#     else:
#         mask_range = np.round(
#             [center-width/2, center+width/2]
#         ).astype(int)
#         mask = np.zeros(npix).astype(bool)
#         mask[mask_range[0]:mask_range[1]] = True
#     return mask

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


