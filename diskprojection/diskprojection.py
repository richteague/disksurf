"""
Combined function to get everything.
"""
from astropy.convolution import convolve, Gaussian1DKernel
from scipy.interpolate import interp1d
from .detect_peaks import detect_peaks
from gofish import imagecube
import numpy as np


class disk_observation(imagecube):
    """
    Wrapper of a GoFish imagecube class.

    Args:
        path (str): Relative path to the FITS cube.
        FOV (optional[float]): Clip the image cube down to a specific
            field-of-view spanning a range ``FOV``, where ``FOV`` is in
            [arcsec].
    """

    def __init__(self, path, FOV=None):
        super().__init__(path=path, FOV=FOV)

    def get_emission_surface(self, inc, PA, x0=0.0, y0=0.0, chans=None,
                             r_min=None, r_max=None, smooth=1.0,
                             return_sorted=True, smooth_threshold_kwargs=None,
                             detect_peaks_kwargs=None):
        """
        Implementation of the method described in Pinte et al. (2018). There
        are several pre-processing options to help with the peak detection.

        Args:
            inc (float): Disk inclination in [degrees].
            PA (float): Disk position angle in [degrees].
            x0 (optional[float]): Disk offset along the x-axis in [arcsec].
            y0 (optional[float]): Disk offset along the y-axis in [arcsec].
            chans (optional[tuple]): First and last channels to include in the
                inference.
            r_min (optional[float]): Minimuim radius in [arcsec] of values to
                return. Default is all possible values.
            r_max (optional[float]): Maximum radius in [arcsec] of values to
                return. Default is all possible values.
            smooth (optional[float]): Prior to detecting peaks, smooth the
                pixel column with a Gaussian kernel with a FWHM equal to
                ``smooth * cube.bmaj``. If ``smooth == 0`` then no smoothing is
                applied.
            return_sorted (optional[bool]): If ``True``, return the points
                ordered in increasing radius.
            smooth_threshold_kwargs (optional[dict]): Keyword arguments passed
                to ``smooth_threshold``.
            detect_peaks_kwargs (optional[dict]): Keyword arguments passed to
                ``detect_peaks``.

        Returns:
            r, z, Fnu, v (arrays): Arrays of radius, height, flux density and
                velocity.
        """

        # Determine the spatial and spectral region to fit.
        r_min = 0.0 if r_min is None else r_min
        r_max = self.xaxis.max() if r_max is None else r_max
        if r_min >= r_max:
            raise ValueError("`r_min` must be less than `r_max`.")
        chans = [0, self.data.shape[0]-1] if chans is None else chans
        if len(chans) != 2:
            raise ValueError("`chans` must be a length 2 list of channels.")
        if chans[1] >= self.data.shape[0]:
            raise ValueError("`chans` extends beyond the number of channels.")
        chans[0], chans[1] = int(min(chans)), int(max(chans))
        data = self.data.copy()[chans[0]:chans[1]+1]
        if self.verbose:
            velo = [self.velax[chans[0]] / 1e3, self.velax[chans[1]] / 1e3]
            velo = [min(velo), max(velo)]
            print("Using {:.2f} km/s to {:.2f} km/s,".format(velo[0], velo[1])
                  + ' and {:.2f}" to {:.2f}".'.format(r_min, r_max))

        # Shift and rotate the data.
        if x0 != 0.0 or y0 != 0.0:
            if self.verbose:
                print("Centering data cube...")
            x0_pix = x0 / self.dpix
            y0_pix = y0 / self.dpix
            data = disk_observation.shift_center(data, x0_pix, y0_pix)
        if PA != 90.0 and PA != 270.0:
            if self.verbose:
                print("Rotating data cube...")
            data = disk_observation.rotate_image(data, PA)

        # Get the masked data.
        if smooth_threshold_kwargs is None:
            smooth_threshold_kwargs = {}
        data = self.radial_threshold(data, inc, **smooth_threshold_kwargs)

        # Default dicionary for kwargs.
        if detect_peaks_kwargs is None:
            detect_peaks_kwargs = {}

        # Define the smoothing kernel.
        if smooth > 0.0:
            kernel = Gaussian1DKernel((smooth * self.bmaj) / self.dpix / 2.235)
        else:
            kernel = None

        # Find all the peaks.
        if self.verbose:
            print("Detecting peaks...")

        peaks = []
        for c_idx in range(data.shape[0]):
            for x_idx in range(data.shape[2]):
                x_c = self.xaxis[x_idx]
                mpd = detect_peaks_kwargs.get('mpd', 0.05 * abs(x_c))
                try:
                    profile = data[c_idx, :, x_idx]
                    if kernel is not None:
                        profile = convolve(profile, kernel, boundary='wrap')
                    y_idx = detect_peaks(profile, mpd=mpd,
                                         **detect_peaks_kwargs)
                    y_idx = y_idx[data[c_idx, y_idx, x_idx].argsort()]
                    y_f, y_n = self.yaxis[y_idx[-2:]]
                    y_c = 0.5 * (y_f + y_n)
                    r = np.hypot(x_c, (y_f - y_c) / np.cos(np.radians(inc)))
                    if not r_min <= r <= r_max:
                        raise ValueError("Out of bounds.")
                    z = y_c / np.sin(np.radians(inc))
                    Fnu = data[c_idx, y_idx[-1], x_idx]
                except (ValueError, IndexError):
                    r, z, Fnu = np.nan, np.nan, np.nan
                peaks += [[r, z, Fnu, self.velax[c_idx]]]
        peaks = np.squeeze(peaks).T
        peaks = peaks[:, np.isfinite(peaks[2])]

        # Sort the values in increasing radius.
        if return_sorted:
            idxs = np.argsort(peaks[0])
            peaks = [p[idxs] for p in peaks]
        return peaks

    def radial_threshold(self, rotated_data, inc, nsigma=1.0, smooth=1.0,
                         think_positively=True, mask_value=0.0):
        """
        Calculates a radial profile of the peak flux density including the mean
        and the azimuthal scatter. The latter defines a threshold for clipping.

        Args:
            rotated_data (ndarray): The data to mask, rotated such that the
                red-shifted axis of the disk aligns with the x-axis (i.e. that
                ``PA == 90`` or ``PA == 270``).
            inc (float): Inclination of the disk in [deg].
            nsigma (optional[float]): Mask all pixels with a flux density less
                than ``mu - nsigma * sig``, where ``mu`` and ``sig`` are
                the radially varying mean and standard deviation of the peak
                flux density.
            smooth (optional[float]): Smooth the radial profiles prior to the
                interpolation with a Gaussian kernal with a FWHM of
                ``smooth * BMAJ``.
            think_positively (optional[bool]): Only consider positive values.
            mask_value (optional[int]): Value to use for masked pixels.

        Returns:
            masked_data (ndarray): A masked verion of ``rotated_data`` where
                all masked values are ``mask_value``.

        """
        if nsigma == 0.0:
            return rotated_data
        Fnu = np.nanmax(rotated_data, axis=0).flatten()
        rbins, rpnts = self.radial_sampling()
        rvals = self.disk_coords(x0=0.0, y0=0.0, inc=inc, PA=0.0)[0]
        ridxs = np.digitize(rvals.flatten(), rbins)

        avgTb, stdTb = [], []
        for r in range(1, rbins.size):
            _tmp = Fnu[ridxs == r]
            _tmp = _tmp[_tmp > 0.0] if think_positively else _tmp
            avgTb += [np.nanmean(_tmp)]
            stdTb += [np.nanstd(_tmp)]
        avgTb = np.array(avgTb)
        stdTb = np.array(stdTb)

        if smooth > 0.0:
            kernel = Gaussian1DKernel((smooth * self.bmaj) / self.dpix / 2.235)
            avgTb = convolve(avgTb, kernel, boundary='wrap')
            stdTb = convolve(stdTb, kernel, boundary='wrap')

        Fnu_clip = interp1d(rpnts, avgTb - nsigma * stdTb,
                            bounds_error=False, fill_value=0.0)(rvals)
        return np.where(rotated_data >= Fnu_clip, rotated_data, mask_value)

    def clip_emission_surface(self, r, z, Fnu, v, min_zr=None, max_zr=None,
                              min_Fnu=None, max_Fnu=None, min_v=None,
                              max_v=None, return_mask=False):
        """
        Clip the emission surface based on simple cuts.

        Args:
            r (array): Array of radius values in [arcsec].
            z (array): Array of height values in [arcsec].
            Fnu (array): Array of flux densities in [Jy/beam].
            v (array): Array of velocities in [m/s].
            min_zr (optional[float]): Mimumum z/r value.
            max_zr (optional[float]): Maximum z/r value.
            min_Fnu (optional[float]): Minimum flux density value in [Jy/beam].
            max_Fnu (optional[float]): Maximum flux density value in [Jy/beam].
            min_v (optional[float]): Minimum velocity in [m/s].
            max_v (optional[float]): Maximum velocity in [m/s].
            return_mask (optional[bool]): If True, just return the mask,
                otherwise return the mask applied to the arrays.

        Returns:
            r, z, Fnu, v (arrays): Arrays of radius, height, flux density and
                velocity.
        """
        # positive r values only
        mask = r > 0.0

        # z/r cuts
        if min_zr is not None:
            mask = mask & (z / r >= min_zr)
        if max_zr is not None:
            mask = mask & (z / r <= max_zr)

        # flux value cuts
        if min_Fnu is not None:
            mask = mask & (Fnu >= min_Fnu)
        if max_Fnu is not None:
            mask = mask * (Fnu <= max_Fnu)

        # velocity range
        if min_v is not None:
            mask = mask & (v >= min_v)
        if max_v is not None:
            mask = mask & (v <= max_v)

        if return_mask:
            return mask
        return r[mask], z[mask], Fnu[mask], v[mask]

    def iterative_clip_emission_surface(self, r, z, Fnu=None, v=None,
                                        nsigma=1.0, niter=3, window=1.0,
                                        min_sigma=0.0, return_mask=False):
        """
        Iteratively clip the emission surface. For a given window (given as a
        function of beam major axis), a running mean, ``mu``, and standard
        deviation, ``sig`` is calculated. All pixels that do not satisfy
        ``abs(z - mu) < nsigma * sig`` are removed. This is performed a
        ``niter`` number of times. To prevent this removing all values a
        ``min_sigma`` value can be provided which sets a minimum value to the
        width of the clipping.

        Args:
            r (array): An array of radius values in [arcsec].
            z (array): An array of corresponding z values in [arcsec].
            Fnu (optional[array]): Array of flux densities in [Jy/beam].
            v (optional[array]): Array of velocities in [m/s].
            nsigma (optional[float]): The number of standard deviations away
                from the running mean to consider a 'good fit'. A larger number
                is more conservative.
            niter (optional[int]): The number of iterations of clipping.
            window (optional[float]): The width of the window as a fraction of
                the beam major axis. This is only rough.
            min_sigma (optional[float]): Minimum standard deviation to use in
                the iterative clipping as a fraction of the beam major axis.
        """
        to_return = [r, z]
        if Fnu is not None:
            to_return += [Fnu]
        if v is not None:
            to_return += [v]
        idxs = np.argsort(to_return[0])
        to_return = [p[idxs] for p in to_return]
        window = int((window * self.bmaj) / np.diff(to_return[0]).mean())
        min_sigma = (min_sigma * self.bmaj)
        mask = disk_observation.sigma_clip(to_return[1], nsigma=nsigma,
                                           niter=niter, window=window,
                                           min_sigma=min_sigma)
        if return_mask:
            return mask
        return [p[mask] for p in to_return]

    def estimate_channel_range(self, average='median', nsigma=5.0, nchan=None):
        """
        Estimate the channel range to use for the emission surface inference.
        This is done by calculating a spectrum based on the average pixel value
        for each channel (using either the median or mean, specified by the
        ``average`` argument), then selecting channels ``nsigma`` times above
        the stanard deviation of the spectrum calculated from the first and
        last ``nchan`` channels.

        .. note::
            Sometime a particularly noisy channel will result in a larger than
            expected channel range. This should not affect the performance of
            ``get_emission_surface`` if appropriate clipping and masking is
            applied afterwards.

        Args:
            average (optional[str]): Type of average to use, either
                ``'median'`` or ``'mean'``.
            nsigma (optional[float]): The RMS factor used to clip channels.
            nchan (optional[int]): The number of first and last channels to use
                to estimate the standard deviation of the spectrum.

        Returns:
            chans (list): A tuple of first and last channels to use for the
                ``chans`` argument in ``get_emission_surface``.

        """
        if average.lower() == 'median':
            avg = np.nanmedian
        elif average.lower() == 'mean':
            avg = np.nanmean
        else:
            raise ValueError("Unknown `average` value: {}.".format(average))
        spectrum = np.array([avg(c) for c in self.data])
        nchan = int(self.velax.size / 3) if nchan is None else int(nchan)
        if nchan > self.velax.size / 2 and self.verbose:
            print("WARNING: `nchan` larger than half the spectral axis.")
        rms = np.nanstd([spectrum[:nchan], spectrum[-nchan:]])
        mask = abs(spectrum) > nsigma * rms
        return [self.channels[mask][0], self.channels[mask][-1] + 1]

    def bin_in_radius(self, r, x, rbins=None, rvals=None, average='mean',
                      uncertainty='std'):
        """
        Radially bin the ``x`` data. The error can either be the standard
        deviation in the bin (``error='std'``), or the 16th to 84th percentiles
        (``error='percentiles'``).
        """
        from scipy.stats import binned_statistic
        rbins, rvals = self.radial_sampling(rbins=rbins, rvals=rvals)

        if average.lower() == 'mean':
            avg_func = np.nanmean
        elif average.lower() == 'median':
            avg_func = np.nanmedian
        else:
            warning = "Unknown `average` value, {}."
            raise ValueError(warning.format(average))
        z_avg = binned_statistic(r, x, bins=rbins, statistic=avg_func)[0]

        if uncertainty.lower() == 'std':
            z_err = binned_statistic(r, x, bins=rbins, statistic=np.nanstd)[0]
        elif uncertainty.lower() == 'percentiles':

            def err_a(x):
                return np.nanpercentile(x, [16.0])

            def err_b(x):
                return np.nanpercentile(x, [50.0])

            def err_c(x):
                return np.nanpercentile(x, [84.0])

            z_err_a = binned_statistic(r, x, bins=rbins, statistic=err_a)[0]
            z_err_b = binned_statistic(r, x, bins=rbins, statistic=err_b)[0]
            z_err_c = binned_statistic(r, x, bins=rbins, statistic=err_c)[0]
            z_err = np.array([z_err_b - z_err_a, z_err_c - z_err_b])
        else:
            warning = "Unknown `uncertainty` value, {}."
            raise ValueError(warning.format(uncertainty))

        return rvals, z_avg, z_err

    @staticmethod
    def rotate_image(data, PA):
        """
        Rotate the image such that the red-shifted axis aligns with the x-axis.

        Args:
            data (ndarray): Data to rotate if not the attached data.
            PA (float): Position angle of the disk, measured to the major axis
                ofthe disk, eastwards (anti-clockwise) from North, in [deg].

        Returns:
            ndarray: Rotated array the same shape as ``data``.
        """
        from scipy.ndimage import rotate
        to_rotate = np.where(np.isfinite(data), data, 0.0)
        PA -= 90.0
        if to_rotate.ndim == 2:
            to_rotate = np.array([to_rotate])
        rotated = np.array([rotate(c, PA, reshape=False) for c in to_rotate])
        if data.ndim == 2:
            rotated = rotated[0]
        return rotated

    @staticmethod
    def shift_center(data, x0, y0):
        """
        Shift the source center by ``x0`` [pix] and ``y0`` [pix] in the `x` and
        `y` directions, respectively.

        Args:
            data (ndarray): Data to shift if not the attached data.
            x0 (float): Shfit along the x-axis in [pix].
            y0 (float): Shifta long the y-axis in [pix].

        Returns:
            ndarray: Shifted array the same shape as ``data``.
        """
        from scipy.ndimage import shift
        to_shift = np.where(np.isfinite(data), data, 0.0)
        if to_shift.ndim == 2:
            to_shift = np.array([to_shift])
        shifted = np.array([shift(c, [-y0, x0]) for c in to_shift])
        if data.ndim == 2:
            shifted = shifted[0]
        return shifted

    @staticmethod
    def rolling_stats(x, window=51):
        """Rolling average and standard deviation."""
        w = window if window % 2 else window + 1
        edge = int((w - 1) / 2)
        xx = np.insert(x, 0, x[0] * np.ones(edge))
        xx = np.insert(xx, -1, x[-1] * np.ones(edge))
        avg = np.squeeze([np.mean(xx[i+edge:i+w+edge]) for i in range(x.size)])
        std = np.squeeze([np.std(xx[i+edge:i+w+edge]) for i in range(x.size)])
        return avg, std

    @staticmethod
    def sigma_clip(x, nsigma=1.0, niter=3, window=51, min_sigma=0.0):
        """Iterative sigma clipping, returns a mask."""
        xtmp = x.copy()
        xnum = np.arange(xtmp.size)
        for n in range(niter):
            mu, sigma = disk_observation.rolling_stats(xtmp, window)
            sigma = np.clip(sigma, a_min=min_sigma, a_max=None)
            mask = abs(xtmp - mu) < nsigma * sigma
            xtmp, xnum = xtmp[mask], xnum[mask]
        return np.squeeze([xx in xnum for xx in np.arange(x.size)])
