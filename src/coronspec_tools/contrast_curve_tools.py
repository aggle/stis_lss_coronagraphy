"""
Tools for generating contrast curves
"""
import warnings

# from pathlib import Path
import pathlib

import numpy as np
from scipy import ndimage
from scipy import interpolate
from scipy import optimize

import matplotlib as mpl
from matplotlib import pyplot as plt

import pandas as pd

from astropy import units
from astropy.wcs import WCS

from coronspec_tools import (
    observing_sequence,
    sdi_tools,
    retrieval_tools, 
    utils as ctutils,
    misc as ctmisc,
)

def sep2row(sep, obs):
    """
    Convert separation in arcsec to a row of the stamp
    """
    row = obs.occ_stamp.wcs.world_to_pixel(obs.wlsol[0], sep+obs.occ_sep)[1]
    return row
def row2sep(row, obs):
    """
    Convert separation in arcsec to a row of the stamp
    """
    sep = obs.occ_stamp.wcs.pixel_to_world(obs.wlsol[0], row)[1]-obs.occ_sep
    return sep

def calc_snr_at_contrast(
    contrast,
    ret : retrieval_tools.Retriever,
    noise_floor : np.ndarray,
    sep : float | units.Quantity,
    wl_range = None,
    spectrum = 1.,
) -> float:
    """
    Inject a spectrum at the given contrast and separation, and return the SNR
    """
    if not isinstance(sep, units.Quantity):
        sep = units.Quantity(sep, unit='arcsec')
    if wl_range is None:
        wl_range = ret.obs.wlsol[[0, -1]].to("nm").value
    snr_cols = np.where(
        (ret.obs.wlsol.to("nm").value >= wl_range[0]) & (ret.obs.wlsol.to("nm").value <= wl_range[1])
    )[0][[0, -1]]
    sep_row = int(sep2row(sep, ret.obs))
    template_array = ret.obs.occ_stamp.data.copy()
    ret.inject_and_process(
        template_array, 
        inj_row = sep_row,
        template_trace = ret.template_trace,
        spectrum = spectrum,
        scale = contrast,
        psf_modeling_args=dict(fit_poly=1, fit_pad=50)
    )
    residual = ret.inj_results.loc[sep_row, 'signal'].value
    meas_snr = np.nanmean((residual/noise_floor[sep_row])[snr_cols[0]:snr_cols[1]])
    return meas_snr
    # return np.abs(target_snr - meas_snr)

def compute_noise_floor(residual, rsize=3, csize=5):
    """Compute the 1-sigma noise floor in a moving box window"""
    noise_floor = np.zeros_like(residual)
    for r in np.arange(residual.shape[0]):
        for c in np.arange(residual.shape[1]):
            rlo = max(0, r-rsize)
            rhi = min(residual.shape[0], r+rsize)+1
            clo = max(0, c-csize)
            chi = min(residual.shape[1], c+csize)+1
            box = residual[rlo:rhi, clo:chi]
            noise_floor[r, c] = np.nanstd(box)
    return noise_floor


def curve_wrapper(
        sx1_file, unocc_file, occ_file, output_folder="contrast_curve_figures",
        spectrum = 1.
):
    """
    A wrapper for computing contrast curves ab initio
    """

    wl_ranges = {
        # 'g430l': (400, 460),
        # 'short': (550, 600),
        # 'medium': (720, 780),
        # 'long': (900, 950),
        'broad': (550, 950),
        # 'narrow': (650, 662)
    }
    snr_thresholds =  (3, 5)
    separations = units.Quantity([0.3, 0.4, 0.5, 0.6, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.75, 3.5], unit='arcsec')

    output_folder = pathlib.Path(output_folder)

    POSTARG = ctmisc.fits.getval(occ_file, 'POSTARG2', 0) * units.arcsec
    
    obs = observing_sequence.ObsSeq(
        sx1_file=sx1_file,
        unocc_file=unocc_file,
        occ_file=occ_file,
        trace_width = 11, # cut out a stamp of this width in rows around the unocculted trace
        occ_stamp_width = 181, # cut a stamp of this width in rows around the occulted star position
        median_clean = 10, # apply a median filter of 2x this width in columns to smooth bad pixels
        contrast = True, # if True, divide by the unocculted spectrum to work in units of contrast
    )
    sdi = sdi_tools.SDI(
        obs,
        ref_wl_ind=-1,
        psf_halfwidth=4,
    )
    sdi.compute_scaled_stamp(sdi.ref_wl_ind, obs.occ_stamp.data, obs.occ_stamp_center)
    # perform initial PSF subtraction
    model_lo = 30
    model_hi = 181-31
    sdi.generate_model_results_df(model_lo, model_hi, model_kwargs=dict(fit_poly=1, fit_pad=50))
    psf_model = sdi.assemble_psf_model_image()
    residual = psf_model - sdi.obs.occ_stamp.data
    noise_floor = compute_noise_floor(residual, rsize=3, csize=5)


    ret = retrieval_tools.Retriever(sdi)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        results = []

        for wl_range in wl_ranges.values():
            snr_cols = np.where(
                (ret.obs.wlsol.to("nm").value >= wl_range[0]) & (ret.obs.wlsol.to("nm").value <= wl_range[1])
            )[0][[0, -1]]
            for snr_thresh in snr_thresholds:
                # above the bar
                for separation in separations - POSTARG:
                    res = optimize.minimize_scalar(
                        lambda contrast: np.abs(
                            snr_thresh - calc_snr_at_contrast(contrast, ret, noise_floor, separation, wl_range, spectrum)
                        ),
                        bounds=[1e-7, 1e-1]
                    )
                    params = dict(target_snr=snr_thresh, separation=separation.value, wl_range=wl_range, contrast=res.x)
                    params['noise'] = np.nanmean(noise_floor[int(sep2row(separation, obs)), snr_cols[0]:snr_cols[1]])
                    results.append(pd.Series(params))
                    print(params)
                # # below the bar
                # for separation in -1*separations - POSTARG:
                #     res = optimize.minimize_scalar(
                #         lambda contrast: np.abs(
                #             snr_thresh - calc_snr_at_contrast(contrast, ret, noise_floor, separation, wl_range, spectrum)
                #         ),
                #         bounds=[1e-7, 1e-1]
                #     )
                #     params = dict(target_snr=snr_thresh, separation=separation.value, wl_range=wl_range, contrast=res.x)
                #     params['noise'] = np.nanmean(noise_floor[int(sep2row(separation, obs)), snr_cols[0]:snr_cols[1]])
                #     results.append(pd.Series(params))
                #     print(params)

        results = pd.DataFrame(results)
        filename = f"{ret.obs.hdrs['occ'][0]['TARGNAME']}_{POSTARG.value}_contrast.csv"
        results.to_csv(output_folder / filename, index=False)

        for wl_range, group in results.groupby("wl_range"):
            fig = plot_curves(pd.DataFrame(group), wl_range, f"{ret.obs.hdrs['occ'][0]['TARGNAME']} | {POSTARG.value}\"")
            fig.savefig(
                output_folder / filename.replace(
                    "contrast",f"{wl_range[0]}-{wl_range[1]}-contrast"
                ).replace(
                    "csv","png"
                ),
                dpi=150
            )


def plot_curves(results_df, wl_range, figtitle=''):
    fig, ax = plt.subplots()
    if figtitle != '':
        fig.suptitle(figtitle)
    ax.set_title(f"Detection limits in region {wl_range} nm")
    # above bar
    for i, (snr, group) in enumerate(results_df.query("separation > 0").groupby("target_snr")):
        index = group.sort_values(by='separation').index
        ax.plot(group.loc[index, "separation"], group.loc[index, "contrast"], label=f"{snr}-sigma", c=f'C{i}')
        ax.plot(group.loc[index, "separation"], group.loc[index, "noise"], c=f'gray')
    # below bar
    for i, (snr, group) in enumerate(results_df.query("separation < 0").groupby("target_snr")):
        index = group.sort_values(by='separation', ascending=False).index
        ax.plot(-1*group["separation"], group["contrast"], ls='--', c=f'C{i}')
        ax.plot(-1*group.loc[index, "separation"], group.loc[index, "noise"], ls='--', c=f'gray')
    if len(results_df.query("separation < 0")) > 0:
        ax.plot([], [], c='k', ls='--', label='below bar')
    ax.plot([], [], c='gray', ls='-', label='noise floor')
        
    ax.legend(title="SNR threshold")
    ax.set_ylabel("Contrast")
    ax.set_xlabel("Separation [arcsec]")
    ax.set_yscale("log")
    return fig
