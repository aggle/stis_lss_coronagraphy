STIS Coronagraphic Spectroscopy Tools
=====================================

Tools for preparing STIS Coronagraphic Spectroscopy

This repo is in a pretty disorganized state at the moment. Here's what works:

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
