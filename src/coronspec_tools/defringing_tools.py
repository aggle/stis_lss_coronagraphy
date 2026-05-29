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
    crj_file : str | Path,
    # flat_file : str | Path,
    # wavecal_file : str | Path,
    # sci_id : str,
    # data_folder : str | Path = '.',
    output_dir : str | Path = '.',
    beg_shift=-0.5,
	end_shift=2,
	shift_step=0.1,
    beg_scale=0.8,
	end_scale=1.7,
	scale_step=0.04,
    compare_spectra : bool = False,
    extraction_row : int | None = None,
):
    """
    Apply fringe flat correction to 2-D spectral images

    Parameters
    ----------
    crj_file : str | Path
      2-D spectral images with fringes that need correcting
    output_dir : str | Path = '.'
      the directory for writing the generated files
    beg_shift=-0.5
	end_shift=2
	shift_step=0.1
    beg_scale=0.8
	end_scale=1.7
	scale_step=0.04
    compare_spectra : bool = False
    """
    # data_folder = Path(data_folder)
    crj_file = Path(crj_file)
    rootname = fits.getval(crj_file, 'ROOTNAME', 0).lower()
    data_folder = crj_file.parent

    mode = fits.getval(crj_file,'OPT_ELEM', 0)
    aper = fits.getval(crj_file,'APERTURE', 0)

    # Normalize the fringe flat
    flat_id = fits.getval(crj_file, 'FRNGFLAT', 0).lower()
    flat_file = data_folder / f"{flat_id}_raw.fits"

    output_dir = Path(output_dir)
    normflat_file =  output_dir /f"{flat_id}_nsp.fits"
    wavecal_file = output_dir / f"{rootname}_wav.fits"
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
        output_fname = output_dir /  infile.name.replace(infile.stem.split("_")[-1], 'dx1d')
        if output_fname.exists():
            os.remove(output_fname)
            print(output_fname, output_fname.exists())
        drj_specfile = extract_spectrum_from_2d(drj_file, str(output_fname), extraction_row)

        if compare_spectra:
            infile = Path(crj_file)
            output_fname = output_dir /  infile.name.replace(infile.stem.split("_")[-1], 'x1d')
            if output_fname.exists():
                os.remove(output_fname)
            crj_specfile = extract_spectrum_from_2d(crj_file, str(output_fname), extraction_row)
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


def extract_spectrum_from_2d(
        file2d : str | Path,
        output_name : str | Path | None = None,
        extraction_row : int | None = None,
):
    """
    file2d: 2d spectral file
    output_name : full path to output 1d spectrum file
    """
    file2d = Path(file2d)
    output_name = Path(output_name)
    with io.capture_output() as captured:
        stistools.x1d.x1d(str(file2d), output=str(output_name), a2center=extraction_row)
    return output_name
