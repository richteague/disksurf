"""
Functions to infer the average height of the emission based on the method
presented in Pinte et al. (2018a).
"""

from detect_peaks import detect_peaks
from scipy.ndimage import rotate, shift
from scipy.interpolate import interp1d


def measure_height(cube, inc, PA, x0=0.0, y0=0.0, chans=None, threshold=0.95):
    """
    Infer the height of the emission surface from the provided cube.

    Args:
        cube (imgcube instance): An imgcube instance of the line data.
        inc (float): Inclination of the source in [degrees].
        PA (float): Position angle of the source in [degrees].
        x0 (optional[float]): Source center offset in x direction in [arcsec].
        y0 (optional[float]): Source center offset in y direction in [arcsec].
        chans (optional[list]): The lower and upper channel numbers to include
            in the fitting.
        threshold (optional[float]): Fraction of the peak intensity
            at that radius to clip.
        smooth (optional[list]):
    """

    # Extract the channels to use.
    if chans is None:
        chans = [0, cube.velax.size]
    chans = np.atleast_1d(chans)
    chans[0] = max(chans[0], 0)
    chans[1] = min(chans[1], cube.velax.size)
    data = cube.data[chans[0]:chans[1]+1]

    # Shift the images to center the image.
    if x0 != 0.0 or y0 != 0.0:
        dy = -y0 / cube.dpix
        dx = x0 / cube.dpix
        data = np.array([shift(c, [dy, dx]) for c in data])

    # Rotate the image so major axis is aligned with x-axis.
    if PA is not None:
        data = np.array([rotate(c, PA - 90.0, reshape=False) for c in data])

    # Make a radial profile of the peak values.
    if threshold > 0.0:
        Tb = np.max(data, axis=0).flatten()
        rvals = cube._get_midplane_polar_coords(0.0, 0.0, inc, 0.0)[0]
        rbins = np.arange(0, cube.xaxis.max() + cube.dpix, cube.dpix)
        ridxs = np.digitize(rvals.flatten(), rbins)
        avgTB = np.array([np.mean(TB[ridxs == r]) for r in range(1, rbins.size)])
        kernel = np.ones(np.ceil(cube.bmaj / cube.dpix).astype('int'))
        avgTB = np.convolve(avgTB, kernel / np.sum(kernel), mode='same')

    # Clip everything below this value.
    avgTB = interp1d(rbins[:-1], threshold * avgTB,
                     fill_value=np.nan, bounds_error=False)
    data = np.where(data >= avgTB(rvals), data, 0.0)

    # Find all the peaks. Save the (r, z, Tb) value. Here we convolve the
    # profile with a top-hat function to reduce some of the noise. We find the
    # two peaks and follow the method from Pinte et al. (2018a) to calculate
    # the height of the emission.
    smooth = smooth / np.sum(smooth) if smooth is not None else [1.0]
    peaks = []
    for c_idx in range(data.shape[0]):
        for x_idx in range(data.shape[2]):
            x_c = cube.xaxis[x_idx]
            mpd = kwargs.pop('mpd', 0.05 * abs(x_c))
            try:
                profile = np.convolve(data[c_idx, :, x_idx],
                                      smooth, mode='same')
                y_idx = detect_peaks(profile, mpd=mpd)
                y_idx = y_idx[data[c_idx, y_idx, x_idx].argsort()]
                y_f, y_n = cube.yaxis[y_idx[-2:]]
                y_c = 0.5 * (y_f + y_n)
                r = np.hypot(x_c, (y_f - y_c) / np.cos(np.radians(inc)))
                z = y_c / np.sin(np.radians(inc))
                if z > 0.5 * r or z < 0:
                    raise ValueError()
                Tb = data[c_idx, y_idx[-1], x_idx]
            except:
                r, z, Tb = np.nan, np.nan, np.nan
            peaks += [[r, z, Tb]]
    return np.squeeze(peaks).T
