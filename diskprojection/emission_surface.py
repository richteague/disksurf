"""
Takes an image cube (as a .fits) and measures the emission surface folling the
approach outlined in Pinte et al. (2018) [1]. This needs to know the disk
inclination and position angle of the source (along with any offsets in the
source center relative to the image center).

If you use the command line verion it will save the results as a Numpy .npz
file as *.emission_height.npy. Alternatively you can import the function which
which has slightly more functionality.

Uses the `detect_peaks` code from Marcos Duarte [2].

References:
    [1] - https://ui.adsabs.harvard.edu/abs/2018A%26A...609A..47P/abstract
    [2] - https://github.com/demotu/BMC
"""

import argparse
import numpy as np
from gofish import imagecube

__all__ = ['get_emission_surface']


def get_emission_surface(path, inc, PA, dx0=0.0, dy0=0.0, chans=None,
                         rmax=None, threshold=0.95, smooth=2,
                         detect_peaks_kwargs=None, verbose=True,
                         auto_correct=True):
    """
    Returns the emission height of the provided image cube following Pinte et
    al. (2018).
    Args:
        path (str): Path to the fits cube.
        inc (float): Inclination of the disk in [deg].
        PA (float): Position angle of the disk in [deg].
        dx0 (Optional[float]): Offset in Right-Ascenion in [arcsec].
        dy0 (Optional[float]): Offset in delincation in [arcsec].
        chans (Optional[list]): A list of the first and last channel to use.
        rmax (Optional[float]) Maximum radius to consider in [arcsec].
        threshold (Optional[float]): At a given radius, remove all pixels below
            ``threshold`` times the azimuthally averaged peak intensity value.
        smooth (Optional[int]): Size of top hat kernel to smooth each column
            prior to finding the peak.
        detect_peaks_kwargs (Optional[dict]): Kwargs for ``detect_peaks``.
        verbose (Opional[bool]): Print out messages.
        auto_correct (Optional[bool]): Will check to see if the disk is the
            correct orientation for the code (rotated such that the far edge of
            the disk is to the top). If ``auto_correct=True``, will include an
            additional rotation of 180 [deg] to correct this.
    Returns:
        list: A list of the midplane radius [arcsec], height [arcsec] and flux
        density [Jy/beam] at that location.
    """
    # Import dependencies.
    from scipy.interpolate import interp1d

    # Load up the image cube.
    cube = imagecube(path, FOV=2.2*rmax)
    rmax = rmax if rmax is not None else cube.xaxis.max()

    # Determine the channels to fit.
    chans = [0, cube.data.shape[0]-1] if chans is None else chans
    if len(chans) != 2:
        raise ValueError("`chans` must be a length 2 list of channels.")
    if chans[1] >= cube.data.shape[0]:
        raise ValueError("`chans` extends beyond the number of channels.")
    chans[0], chans[1] = int(min(chans)), int(max(chans))
    data = cube.data[chans[0]:chans[1]+1]
    if verbose:
        velo = [cube.velax[chans[0]] / 1e3, cube.velax[chans[1]] / 1e3]
        print("Using {:.2f} km/s to {:.2f} km/s,".format(min(velo), max(velo))
              + ' and 0" to {:.2f}".'.format(rmax))

    # Shift and rotate the data.
    if dx0 != 0.0 or dy0 != 0.0:
        if verbose:
            print("Centering data cube...")
        data = shift_center(data, dx0 / cube.dpix, dy0 / cube.dpix)
    if verbose:
        print("Rotating data cube...")
    data = rotate_image(data, PA)

    # Check to see if the disk is tilted correctly.
    Tb = np.max(data, axis=0)
    median_top = np.nanmedian(Tb[int(Tb.shape[0] / 2):])
    median_bottom = np.nanmedian(Tb[:int(Tb.shape[0] / 2)])

    if median_bottom > median_top:
        if verbose:
            print("WARNING: " +
                  "The far side of the disk looks like it is to the south.")
        if auto_correct:
            if verbose:
                print("\t Rotating the disk by 180 degrees...\n" +
                      "\t To ignore this step, use `auto_correct=False`.")
            data = data[:, ::-1, ::-1]

    # Make a (smoothed) radial profile of the peak values and clip values.
    if verbose:
        print("Removing low SNR pixels...")
    rbins, _ = cube.radial_sampling()
    rvals = cube.disk_coords(x0=0.0, y0=0.0, inc=inc, PA=0.0)[0]
    Tb = Tb.flatten()
    ridxs = np.digitize(rvals.flatten(), rbins)
    avgTb = []
    for r in range(1, rbins.size):
        avgTb_tmp = Tb[ridxs == r]
        avgTb += [np.mean(avgTb_tmp[avgTb_tmp > 0.0])]
    kernel = np.ones(np.ceil(cube.bmaj / cube.dpix).astype('int'))
    avgTb = np.convolve(avgTb, kernel / np.sum(kernel), mode='same')
    avgTb = interp1d(np.average([rbins[1:], rbins[:-1]], axis=0),
                     threshold * avgTb, fill_value=np.nan,
                     bounds_error=False)(rvals)
    data = np.where(data >= np.where(np.isfinite(avgTb), avgTb, 0.), data, 0.)

    # We convolve the profile to reduce some of the noise.
    smooth = np.ones(smooth) / smooth if smooth is not None else [1.0]

    # Default dicionary for kwargs.
    if detect_peaks_kwargs is None:
        detect_peaks_kwargs = {}

    # Find all the peaks and save the (r, z, Tb) value.
    peaks = []
    if verbose:
        print("Detecting peaks...")
    for c_idx in range(data.shape[0]):
        for x_idx in range(data.shape[2]):
            x_c = cube.xaxis[x_idx]
            mpd = detect_peaks_kwargs.get('mpd', 0.05 * abs(x_c))
            try:
                profile = np.convolve(data[c_idx, :, x_idx],
                                      smooth, mode='same')
                y_idx = detect_peaks(profile, mpd=mpd, **detect_peaks_kwargs)
                y_idx = y_idx[data[c_idx, y_idx, x_idx].argsort()]
                y_f, y_n = cube.yaxis[y_idx[-2:]]
                y_c = 0.5 * (y_f + y_n)
                r = np.hypot(x_c, (y_f - y_c) / np.cos(np.radians(inc)))
                z = y_c / np.sin(np.radians(inc))
                Tb = data[c_idx, y_idx[-1], x_idx]
                v_idx = c_idx
            except (ValueError, IndexError):
                r, z, Tb, v_idx = np.nan, np.nan, np.nan, np.nan
            peaks += [[r, z, Tb, v_idx]]
    peaks = np.squeeze(peaks)

    # Remove any NaN values and sort them.
    r, z, Tb, v_idx = peaks[~np.any(np.isnan(peaks), axis=1)].T
    idx = np.argsort(r)
    r, z, Tb, v_idx = r[idx], z[idx], Tb[idx], v_idx[idx]
    idx = r <= rmax
    if verbose:
        print("Done!\n")
    return r[idx], z[idx], Tb[idx], v_idx[idx]


def rotate_image(data, PA):
    """
    Rotate the image such that the red-shifted axis aligns with the x-axis.
    Args:
        data (ndarray): Data to rotate if not the attached data.
        PA (float): Position angle of the disk, measured to the major axis of
            the disk, eastwards (anti-clockwise) from North, in [deg].
    Returns:
        ndarray: Rotated array the same shape as ``data``.
    """
    from scipy.ndimage import rotate
    to_rotate = np.where(np.isfinite(data), data, 0.0)
    if to_rotate.ndim == 2:
        to_rotate = np.array([to_rotate])
    rotated = np.array([rotate(c, PA - 90., reshape=False) for c in to_rotate])
    if data.ndim == 2:
        rotated = rotated[0]
    return rotated


def shift_center(data, dx0, dy0):
    """
    Shift the source center by ``dx0`` [pix] and ``dy0`` [pix] in the
    x- and y-directions, respectively.
    Args:
        data (ndarray): Data to shift if not the attached data.
        dx0 (float): Shfit along the x-axis in [pix].
        dy0 (float): Shifta long the y-axis in [pix].
    Returns:
        ndarray: Shifted array the same shape as ``data``.
    """
    from scipy.ndimage import shift
    to_shift = np.where(np.isfinite(data), data, 0.0)
    if to_shift.ndim == 2:
        to_shift = np.array([to_shift])
    shifted = np.array([shift(c, [-dy0, dx0]) for c in to_shift])
    if data.ndim == 2:
        shifted = shifted[0]
    return shifted


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False):

    """Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    The function can handle NaN's
    See this IPython Notebook [1]_.
    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/
                                        blob/master/notebooks/DetectPeaks.ipynb
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            m1 = np.hstack((dx, 0)) <= 0
            m2 = np.hstack((0, dx)) > 0
            ire = np.where(m1 & m2)[0]
        if edge.lower() in ['falling', 'both']:
            m1 = np.hstack((dx, 0)) < 0
            m2 = np.hstack((0, dx)) >= 0
            ife = np.where(m1 & m2)[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        thing = np.unique(np.hstack((indnan, indnan-1, indnan+1)))
        ind = ind[np.in1d(ind, thing, invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])
    return ind


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        help='Path to the FITS cube.',
                        type=str)
    parser.add_argument('-inc',
                        help='Inclination of the disk in [deg].',
                        type=float)
    parser.add_argument('-PA',
                        help='Position angle of the disk in [deg].',
                        type=float)
    parser.add_argument('-dx0',
                        help='Offset along the x-axis in [arcsec].',
                        type=float,
                        default=0.0)
    parser.add_argument('-dy0',
                        help='Offset along the y-axis in [arcsec].',
                        type=float,
                        default=0.0)
    parser.add_argument('-chans',
                        help='First and last channel to include in the fit.',
                        type=int,
                        default=None,
                        nargs='+')
    parser.add_argument('-rmax',
                        help='Maximum out radius to consider in [arcsec].',
                        type=float,
                        default=None)
    parser.add_argument('-threshold',
                        help=('Relative fraction of the average peak intensity'
                              + ' to clip below.'),
                        type=float,
                        default=1.0)
    parser.add_argument('--auto_correct',
                        help='Detect if the cube is oriented correctly.',
                        action='store_true')
    args = parser.parse_args()
    if args.chans is not None:
        if len(args.chans) != 2:
            raise ValueError("Must provide the first and last channel only.")
    np.save(args.path.replace('.fits', '.emission_height.npy'),
            get_emission_surface(**vars(args)))
