"""
Apply fringe flat correction to flt, crj, and sx2 files
See https://stistools.readthedocs.io/en/latest/defringe_guide.html
"""
import shutil, os
from pathlib import Path

from astropy.io import fits
import stistools

def defringe_raw(
    sci_file : str | Path,
    flat_file : str | Path,
    wavecal_file : str | Path,
    output_dir : str | Path = '.',
):
    """
    Apply fringe flat correction

    Parameters
    ----------
    crj_file : str | Path
      2-D spectral images with fringes that need correcting
    flat_file : str | Path
      2-D fringe flat image to use for correcting fringes
    """
    mode = fits.getval(sci_file,'OPT_ELEM', 0)
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
        beg_shift=-0.5, end_shift=2, shift_step=0.1,
        beg_scale=0.8, end_scale=1.7, scale_step=0.04
    )
    drj_file = stistools.defringe.defringe(
        str(crj_file), str(frr_file), overwrite=True
    )
    return drj_file

