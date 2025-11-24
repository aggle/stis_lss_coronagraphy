STIS Coronagraphic Spectroscopy Tools
=====================================

Tools for preparing STIS Coronagraphic Spectroscopy

This repo is in a pretty disorganized state at the moment. The notebook `notebooks.Example.ipynb` should hopefully be an up-to-date guide on the basics of what is currently working. Here's what works:

- `coronspec_tools::find_star::find_star_from_wcs()`
  - three arguments:
    - extracted 1-D spectrum
    - 2-D unocculted spectral image
    - 2-D occulted spectral image

  The observing sequence for this mode is to take an unocculted exposure at the
  F1 position and then an occulted exposure at the E1 position, along the slit.
  This function uses the WCS information to find the position of the star in the
  occulted image. It assumes that the HST slews are very accurate. It measures
  the difference between the actual location of the trace in the unocculted 2-D
  image, and the nominal position of the trace in the WCS at coordinates
  (wl_min, 0 deg). Assuming the offset is preserved in the slew between fiducial positions, it applies the measured offset to the nominal position in the occulted exposure. 

- `ObservingSequence` Objects
  - three arguments (same as above)
    - extracted 1-D spectrum
    - 2-D unocculted spectral image
    - 2-D occulted spectral image
  
  `coronspec_tools` uses ObservingSequence objects to keep related files together.
  Specifically, some observing sequence includes a TA image, an unocculted
  exposure, and an occulted exposure. The hstcal pipeline also generates a 1-D
  spectrum from the unocculted trace that is useful in processing the exposures.
  ObservingSequence objects use these files to derive the row position of the
  star in the occulted exposures and prepare the PSF subtraction. 
  
  - Attributes:
    - `_files` : a dict that keeps a record of the initializing files
    - `wlsol`: the wavelength solution, in meters
    - `primary_spectrum`: the point source spectrum, in units of counts/sec
    - `primary_spectrum_unc`: the associated uncertainty from the `ERR` column
    - `unocc_wcs`: WCS object for the unocculted observation
    - `unocc_img`: 2-D spectral image of the unocculted
    - `offset`: distance in degrees of the unocculted point source from the nominal position
    - `unocc_row`: 0-indexed row coordinate corresponding to the offset in the unocculted exposure
    - `unocc_trace`: astropy.nddata.Cutout2D crop of the spectral trace; used for injection and recovery tests
    - `occ_wcs`: WCS object for the occulted observation
    - `occ_img`: 2-D spectral image of the occulted observation
    - `occ_row`: 0-indexed row coordinate corresponding to the offset in the occulted exposure, aka where the star is
