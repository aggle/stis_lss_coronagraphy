"""Various helper functions"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from astropy.io import fits
from astropy import units
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS


def organize_files_by_header(files, ext=0, add_filepath=True, extra_kwds={}):
    """
    Take a bunch of FITS files and combine their headers into a dataframe for sorting

    Parameters
    ----------
    files: list of strings or pathlib.Paths
      list of paths to fits files from jwst
    ext : int [0]
      number of the extension (0=PRI, 1=SCI, 2=ERR, etc)
    add_filepath: bool [True]
      if True, add a column called 'filepath' with the full path to the file
    extra_kwds : {}
      any extra keywords to read from other headers. Format: {hdr: [list of keywords]}

    Output
    ------
    hdr_df : pd.DataFrame
      dataframe of all the file headers

    """
    # make a list of headers and format them
    hdrs = []
    for f in files:
        hdr = fits.getheader(str(f), ext)
        for hdr in extra_kwds:
            for k in extra_kwds[hdr]:
                try:
                    hdr[k] = fits.getval(str(f), k, hdr)
                except KeyError:
                    hdr[k] = 'Error: Key not found'
        hdr = pd.Series(hdr)
        # add the root folder
        hdr['path'] = str(Path(f).parent)
        hdr['filestem'] = '_'.join(Path(f).stem.split('_')[:-1])
        if add_filepath == True:
            hdr['filepath'] = str(Path(f).absolute())
        # drop duplicated index entries, usually this is "''"" and "COMMENT"
        drop_index = hdr.index[hdr.index.duplicated()]
        hdr.drop(index=drop_index, inplace=True)
        # also drop all instances of "''" and 'COMMENT'
        for label in ['','COMMENT']:
            try:
                hdr.drop(labels=label)
            except KeyError:
                # probably this means there are no entries with this index label
                pass
        hdrs.append(hdr)
    hdr_df = pd.concat(hdrs, axis=1).T
    # make a column to preserve the unique file ID in the database
    hdr_df.reset_index(inplace=True)
    hdr_df.rename(columns={'index': 'db_id'}, inplace=True)

    return hdr_df


def print_columns(list_to_print, ncols=5, sort=False):
    """
    Print a list of items in columns.

    Parameters
    ----------
    list_to_print : list
      a list-like object of items to print. All members will be converted to strings.
    ncols : int [5]
      the number of columns
    sort : bool [False]
      if True, sort list alphabetically

    Output
    ------
    no return value; prints to screen

    """
    list_to_print = [str(i) for i in list_to_print]
    if sort == True:
        list_to_print = sorted(list_to_print)
    spacing = max([len(i) for i in list_to_print])
    nitems = len(list_to_print)
    nrows = int(nitems/ncols)
    if nitems%ncols > 0:
        nrows += 1 # this is equivalent to np.ceil

    for i in range(nrows):
        print(' '.join(f"{i:{spacing+2}s}" for i in list_to_print[i::nrows]))


def load_all_files(
        parent_folder : str | Path = '../data/HST'
) -> dict[str, pd.DataFrame] :
    """
    We know what the files are for this program. This method organizes them for
    you and returns them in a dictionary.

    Parameters
    ----------
    parent_folder : str | Path
      the parent folder that has the files

    Output
    ------
    file_managers : dict
      a dict whose keys are the file types and whose entries are file manager
      dataframes

    """
    data_files = sorted(Path(parent_folder).glob("*/*fits"))
    file_managers = {}
    for f in data_files:
        ftype = f.stem.split("_")[1]
        if ftype not in file_managers.keys():
            file_managers[ftype] = []
        file_managers[ftype].append(f)
    for ft in file_managers:
        file_managers[ft] = organize_files_by_header(file_managers[ft])
    return file_managers

def load_wlsol(
        sx1_file : str | Path,
) -> units.Quantity:
    """Given an sx1-type file, load the wavelength solution"""
    hdr = fits.getheader(sx1_file, 1)
    fieldname = hdr['TTYPE3']
    unit = hdr['TUNIT3'][:-1]  # Angstroms, drop the final s
    data = fits.getdata(sx1_file, 1)
    wlsol = units.Quantity(np.squeeze(data[fieldname]), unit=unit.lower())
    return wlsol
    # return np.squeeze(data["WAVELENGTH"])


def make_sx2_mask(
        hdulist : fits.HDUList,
        dq : bool = False,
        mask_edges : bool = True,
        unmask_center_only : bool = False,
) -> np.array:
    """
    Make a mask around the central region of an sx2 file to focus only on the stellar speckle streaks

    Parameters
    ----------
    hdulist : fits.HDUList
      an hdulist of the file that needs a mask
    dq : bool [False]
      whether or not to include the DQ image in the mask
    mask_edges : bool [True]
      whether or not to mask the padded edges of the array
    unmask_center_only : bool [False]
      whether or not to mask everything except a narrow region around the center

    Output
    ------
    mask : np.array
      boolean mask the same size as the image, that leaves only the speckle-fitting region
    """
    # start out with everything masked
    imshape = [i for i in hdulist['SCI'].header['NAXIS[0-9]*'].values()]
    mask = np.tile(False, tuple(imshape))

    if unmask_center_only:
        mask = np.tile([True], hdulist[1].data.shape)
        # unmask the central strip
        mask[530:640] = False

    # mask out the flagged pixels, if requested
    if dq == True:
        dqimg = hdulist[3].data
        mask[np.where(dqimg != 0)] = True

    return mask

def interp_peak(
        hdulist : fits.HDUList,
        search_col : int = 200,
) -> float :
    """
    Guess the position of the star

    Parameters
    ----------
    hdulist : fits.HDUList,
      HDUList with the data in the "SCI" header
    search_col : int = 200,

    Output
    ------
    Define your output

    """
    pass # insert body here
    is_str = isinstance(hdulist, str)
    if is_str:
        hdulist = fits.open(hdulist)
    # initial guess
    star_row = float(hdulist['SCI'].header['CRPIX2'])-1
    img = hdulist['SCI'].data

    interp_rows = np.arange(star_row-20, star_row+20+1, dtype=int)
    func = interp1d(interp_rows, -img[interp_rows, search_col], 'cubic')
    interp_min = minimize_scalar(func, (float(interp_rows[0]), star_row, float(interp_rows[-1])))
    peak_row = interp_min.x
    if is_str:
        hdulist.close()
    return peak_row


# def clean_image(

#         img : np.ndarray,
#         window_size : int = 3,
# ) -> np.ndarray :
#     """
#     Identify outliers and mask them

#     Parameters
#     ----------
#     img : np.ndarray
#       2-d image
#     window_size : int = 3,
#       box of radius 2*window_size + 1

#     Output
#     ------
#     masked_img
#       Return an image 

#     """
#     mask = np.zeros_like(img, dtype=bool)
#     ygrid, xgrid = np.mgrid[:img.shape[0], :img.shape[1]]
#     for y, x in zip(ygrid.ravel(), xgrid.ravel()):
#         ylo = max(0, y-window_size)
#         yhi = min(y+window_size+1, img.shape[0])
#         xlo = max(0, x-window_size)
#         xhi = min(x+window_size+1, img.shape[1])
#         window = img[ylo:yhi,xlo:xhi]
#         # sigma = sigma_clipped_stats(window)[-1]
#         sigma = np.nanstd(window)
#         if img[y, x] > 5*sigma:
#             mask[y, x] = True
#     return mask

def rolling_median(array, window=1):
    median_filtered = np.empty(array.size-2*window)
    for i in np.arange(window, array.size-window):
        median_filtered[i-window] = np.nanmedian(array[i-window:i+window+1])
    return median_filtered
def median_filter_image(img, window=5):
    filtered_img = np.zeros((img.shape[0], img.shape[1]-2*window)) * np.nan
    for row in np.arange(filtered_img.shape[0]):
        filtered_img[row] = rolling_median(img[row], window)
    return filtered_img
