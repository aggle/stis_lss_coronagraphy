"""
Apply fringe flat correction to flt, crj, and sx2 files
See https://stistools.readthedocs.io/en/latest/defringe_guide.html
"""
import shutil, os
from pathlib import Path

import matplotlib as mpl
from matplotlib import pyplot as plt

from astropy.io import fits
import stistools

from IPython.utils import io

def defringe_raw2d(
    sci_file : str | Path,
    flat_file : str | Path,
    wavecal_file : str | Path,
    output_dir : str | Path = '.',
    beg_shift=-0.5,
	end_shift=2,
	shift_step=0.1,
    beg_scale=0.8,
	end_scale=1.7,
	scale_step=0.04,
    compare_spectra : bool = False,
):
    """
    Apply fringe flat correction to 2-D spectral images

    Parameters
    ----------
    crj_file : str | Path
      2-D spectral images with fringes that need correcting
    flat_file : str | Path
      2-D fringe flat image to use for correcting fringes
    """
    mode = fits.getval(sci_file,'OPT_ELEM', 0)
    aper = fits.getval(sci_file,'APERTURE', 0)
    # Normalize the fringe flat
    sci_file = Path(sci_file)
    flat_file = Path(flat_file)
    wavecal_file = Path(wavecal_file)
    output_dir = Path(output_dir)
    normflat_file = output_dir / (flat_file.name.replace("raw","nsp"))
    stistools.defringe.normspflat(
        str(flat_file),
        str(normflat_file),
        wavecal=str(wavecal_file),
        do_cal=True,
    )
    # Flatten the blue end of the flat-field image [ONLY FOR G750L]
    if mode == 'G750L':
        with fits.open(normflat_file, mode='update') as hdulist:
            hdulist[1].data[:,:250] = 1

    # Run prepspec to make a crj file
    with io.capture_output() as captured:
        stistools.defringe.prepspec(str(sci_file), outroot=str(output_dir))
    crj_file = output_dir / sci_file.name.replace("raw","crj")

    # find the optimal scaling to make the fringe flat
    frr_file = Path(str(normflat_file).replace("nsp","frr"))
    if frr_file.exists():
        os.remove(frr_file)
    stistools.defringe.mkfringeflat(
        str(crj_file),
        str(normflat_file),
        str(frr_file),
        beg_shift=beg_shift,
        end_shift=end_shift,
        shift_step=shift_step,
        beg_scale=beg_scale,
        end_scale=end_scale,
        scale_step=scale_step,
    )
    drj_file = stistools.defringe.defringe(
        str(crj_file), str(frr_file), overwrite=True
    )
    if aper == '52X0.2':
        infile = Path(drj_file)
        output_path = Path(".") /  infile.name.replace(infile.stem.split("_")[-1], 'dx1d')
        drj_specfile = extract_spectrum(drj_file, str(output_path))

        if compare_spectra:
            infile = Path(crj_file)
            output_path = Path(".") /  infile.name.replace(infile.stem.split("_")[-1], 'x1d')
            crj_specfile = extract_spectrum(crj_file, str(output_path))

            # Plot both the fringed and the defringed 1D extracted spectra together
            dx1d = fits.open(drj_specfile)
            x1d = fits.open(crj_specfile)
            fig, axes = plt.subplots(
                nrows=2, ncols=1, figsize=(10,7),dpi=150, height_ratios=(3, 1),
                sharex=True
            )
            ax = axes[0]
            ax.plot(dx1d[1].data['WAVELENGTH'][0], dx1d[1].data['FLUX'][0],'-', label='Defringed', alpha=0.7)
            ax.plot(x1d[1].data['WAVELENGTH'][0], x1d[1].data['FLUX'][0],'-', label='Fringed', alpha=0.7)
            ax.grid(visible=True)
            ax.legend()

            ax = axes[1]
            ax.plot(
                dx1d[1].data['WAVELENGTH'][0], dx1d[1].data['FLUX'][0]/x1d[1].data['FLUX'][0],
                '-', c='k'
            )
            ax.set_title("defringed / fringed")
            fig.tight_layout()
            fig.suptitle(fits.getval(drj_file, 'ROOTNAME', 0))
    return drj_file


def extract_spectrum(
        file2d : str | Path,
        output_dir : str | Path,
):
    """
    """
    file2d = Path(file2d)
    output_path = Path(output_dir)
    # output_dir = Path(output_dir)
    # ftype = file2d.stem.split("_")[-1]
    # output_path = output_dir /  file2d.name.replace(ftype, 'x1d')
    with io.capture_output() as captured:
        stistools.x1d.x1d(str(file2d), output=str(output_path))
    return output_path
