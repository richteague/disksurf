"""
A wrapper class to allow a user to extract an emission surface from a datacube
of molecular line emission. This uses the method presented in Pinte et al.
(2018).
"""
from astropy.convolution import convolve, Gaussian1DKernel
from scipy.interpolate import interp1d
from .detect_peaks import detect_peaks
import matplotlib.pyplot as plt
from .surface import surface
from gofish import imagecube
import numpy as np


class observation(imagecube):
    """
    Wrapper of a GoFish imagecube class.

    Args:
        path (str): Relative path to the FITS cube.
        FOV (optional[float]): Clip the image cube down to a specific
            field-of-view spanning a range ``FOV``, where ``FOV`` is in
            [arcsec].
    """

    def __init__(self, path, FOV=None, velocity_range=None):
        super().__init__(path=path, FOV=FOV, velocity_range=velocity_range)
        self.data_aligned_rotated = {}

    def get_emission_surface(self, inc, PA, x0=0.0, y0=0.0, chans=None,
                             r_min=None, r_max=None, smooth=None, nsigma=None,
                             min_SNR=5, detect_peaks_kwargs=None):
        """
        Implementation of the method described in Pinte et al. (2018). There
        are several pre-processing options to help with the peak detection.

        Args:
            inc (float): Disk inclination in [degrees].
            PA (float): Disk position angle in [degrees].
            x0 (optional[float]): Disk offset along the x-axis in [arcsec].
            y0 (optional[float]): Disk offset along the y-axis in [arcsec].
            chans (optional[list]): First and last channels to include in the
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
            r, z, Inu, v (arrays): Arrays of radius, height, flux density and
                velocity.
        """

        # Remove bad inclination:

        if inc == 0.0:
            raise ValueError("Cannot infer height with face on disk.")
        if self.verbose and abs(inc) < 10.0:
            print("WARNING: Inferences with close to face on disk are poor.")

        # Determine the spatial and spectral region to fit.

        r_min = r_min or 0.0
        r_max = r_max or self.xaxis.max()
        if r_min >= r_max:
            raise ValueError("`r_min` must be less than `r_max`.")

        chans, data = self._get_velocity_clip_data(chans=chans)

        # Align and rotate the data such that the major axis is parallel with
        # the x-axis. We can save this as a copy for user later for plotting
        # or repeated surface extractions.

        key = (x0, y0, PA, chans.min(), chans.max())
        try:
            data = self.data_aligned_rotated[key]
        except KeyError:
            data = self._align_and_rotate_data(data=data, x0=x0, y0=y0, PA=PA)
            self.data_aligned_rotated[key] = data

        # Define the smoothing kernel.

        if smooth or 0.0 > 0.0:
            kernel = np.hanning(2.0 * smooth * self.bmaj / self.dpix)
            kernel /= np.sum(kernel)
        else:
            kernel = None

        # Find all the peaks.

        if self.verbose:
            print("Detecting peaks...")
        _surface = self._detect_peaks(data=data, inc=inc, r_min=r_min,
                                      r_max=r_max, chans=chans, kernel=kernel,
                                      min_SNR=min_SNR,
                                      detect_peaks_kwargs=detect_peaks_kwargs)
        return surface(*_surface, chans=chans, rms=self.estimate_RMS(),
                       x0=x0, y0=y0, inc=inc, PA=PA, r_min=r_min, r_max=r_max)

    # -- DATA MANIPULATION -- #

    def _get_velocity_clip_data(self, chans=None):
        """Clip the data based on a provided channel range."""
        chans = chans or [0, self.data.shape[0] - 1]
        chans = np.atleast_2d(chans).astype('int')
        if chans.min() < 0:
            raise ValueError("`chans` has negative values.")
        if chans.max() >= self.data.shape[0]:
            raise ValueError("`chans` extends beyond the number of channels.")
        return chans, self.data.copy()[chans.min():chans.max()+1]

    def _align_and_rotate_data(self, data, x0=None, y0=None, PA=None):
        """Align and rotate the data."""
        if x0 != 0.0 or y0 != 0.0:
            if self.verbose:
                print("Centering data cube...")
            x0_pix = x0 / self.dpix
            y0_pix = y0 / self.dpix
            data = observation.shift_center(data, x0_pix, y0_pix)
        if PA != 90.0 and PA != 270.0:
            if self.verbose:
                print("Rotating data cube...")
            data = observation.rotate_image(data, PA)
        return data

    def _detect_peaks(self, data, inc, r_min, r_max, chans, min_SNR=5.0,
                      kernel=None, return_back=True, detect_peaks_kwargs=None):
        """Wrapper for `detect_peaks.py`."""

        inc_rad = np.radians(inc)
        detect_peaks_kwargs = detect_peaks_kwargs or {}

        # Infer the correct range in the x direction.

        x_idx_max = abs(self.xaxis + r_max).argmin() + 1
        x_idx_min = abs(self.xaxis - r_max).argmin()
        assert x_idx_min < x_idx_max

        # Infer the correct range in the y direction.

        r_max_inc = r_max * np.cos(inc_rad)
        y_idx_min = abs(self.yaxis + r_max_inc).argmin()
        y_idx_max = abs(self.yaxis - r_max_inc).argmin() + 1
        assert y_idx_min < y_idx_max

        # Estimate the noise to remove low SNR pixels.

        min_SNR = min_SNR or -1e10
        min_Inu = min_SNR * self.estimate_RMS()

        # Loop through each channel, then each vertical pixel column to extract
        # the peaks.

        _surface = []
        for c_idx in range(data.shape[0]):

            # Check that the channel is in one of the channel ranges. If not,
            # we skip to the next channel index.

            c_idx_tot = c_idx + chans.min()
            if not any([ct[0] <= c_idx_tot <= ct[1] for ct in chans]):
                continue

            for x_idx in range(x_idx_min, x_idx_max):

                x_c = self.xaxis[x_idx]
                mpd = detect_peaks_kwargs.get('mpd', 0.05 * abs(x_c))
                v = self.velax[c_idx_tot]

                try:

                    # Grab the appropriate column of pixels and optionally
                    # smooth them with a Hanning convolution.

                    cut = data[c_idx, y_idx_min:y_idx_max, x_idx]
                    if kernel is not None:
                        cut_a = np.convolve(cut, kernel, mode='same')
                        cut_b = np.convolve(cut[::-1], kernel, mode='same')
                        cut = np.mean([cut_a, cut_b[::-1]], axis=0)

                    # Returns an array of all the peaks found in the cut and
                    # sort them into order of increasing intensity. Then split
                    # these into those above and below the major axis.

                    y_idx = detect_peaks(cut, mpd=mpd, **detect_peaks_kwargs)
                    y_idx += y_idx_min
                    y_idx = y_idx[data[c_idx, y_idx, x_idx].argsort()]

                    # Check that the two peaks have a minimum SNR value.

                    if min(data[c_idx, y_idx[-2:], x_idx]) < min_Inu:
                        raise ValueError("Out of bounds (RMS).")

                    y_n, y_f = sorted(self.yaxis[y_idx[-2:]])

                    # Remove points that are on the same side of the major
                    # axis of the disk. This may remove poinst in the outer
                    # disk, but that's more conservative anyway.

                    if y_f * y_n > 0.0:
                        raise ValueError("Out of bounds (major axis).")

                    # Calculate the deprojection, making sure the radius is
                    # still in the bounds of acceptable values.

                    y_c = 0.5 * (y_f + y_n)
                    r = np.hypot(x_c, (y_f - y_c) / np.cos(inc_rad))
                    if not r_min <= r <= r_max:
                        raise ValueError("Out of bounds(r).")
                    z = y_c / np.sin(inc_rad)

                    # Include the intensity of the peak position.

                    Inu = data[c_idx, y_idx[-1], x_idx]
                    if np.isnan(Inu):
                        raise ValueError("Out of bounds (Inu).")

                    # Include the back side of the disk, otherwise populate
                    # all associated variables with NaNs. Follow exactly the
                    # same procedure as the front side of the disk.
                    # TODO: Is there a nicer way to replace this chunk of code?

                    try:
                        if min(data[c_idx, y_idx[-4:], x_idx]) < min_Inu:
                            raise ValueError("Out of bounds (RMS).")
                        y_nb, y_fb = sorted(self.yaxis[y_idx[-4:-2]])
                        if y_fb * y_nb > 0.0:
                            raise ValueError("Out of bounds (major axis).")
                        y_cb = 0.5 * (y_fb + y_nb)
                        rb = np.hypot(x_c, (y_fb - y_cb) / np.cos(inc_rad))
                        if not r_min <= rb <= r_max:
                            raise ValueError("Out of bounds (r).")
                        zb = y_cb / np.sin(inc_rad)
                        Inub = data[c_idx, y_idx[-3], x_idx]
                        if np.isnan(Inub):
                            raise ValueError("Out of bounds (Inu).")

                    except (ValueError, IndexError):
                        y_nb, y_fb = np.nan, np.nan
                        rb, zb = np.nan, np.nan
                        Inub = np.nan

                except (ValueError, IndexError):
                    y_n, y_f = np.nan, np.nan
                    r, z = np.nan, np.nan
                    Inu = np.nan
                    y_nb, y_fb = np.nan, np.nan
                    rb, zb = np.nan, np.nan
                    Inub = np.nan

                peaks = [r, z, Inu, v, x_c, y_n, y_f, rb, zb, Inub, y_nb, y_fb]
                _surface += [peaks]

        # Remove any non-finite values and return.

        _surface = np.squeeze(_surface).T
        return _surface[:, np.isfinite(_surface[2])]

    def quick_peak_profile(self, inc, PA, data=None):
        """
        Returns a quick and dirty radial profile of the peak flux density. This
        function does not consider any flared emission surfaces, offset and
        only takes the maximum value along the spectral axis.

        Args:
            inc (float): Disk inclination in [degrees].
            PA (float): Disk position angle in [degrees].
            data (optional[array]): Data to make a profile of. If no data is
                provided, take the maximum of ``self.data`` along the spectral
                axis.

        Returns:
            r, Inu, dInu (array, array, array): Arrays of the peak flux
                density, ``Inu`` at radial positions ``r``. ``dInu`` is given
                by the standard error on the mean.
        """
        data = np.nanmax(self.data.copy(), axis=0) if data is None else data
        if data.ndim != 2:
            raise ValueError("`data` must be a 2D array.")
        data = data.flatten()
        rbins, rpnts = self.radial_sampling()
        rvals = self.disk_coords(x0=0.0, y0=0.0, inc=inc, PA=PA)[0]
        ridxs = np.digitize(rvals.flatten(), rbins)
        Inu, dInu = [], []
        for r in range(1, rbins.size):
            _tmp = data[ridxs == r]
            _tmp = _tmp[np.isfinite(_tmp)]
            Inu += [np.mean(_tmp)]
            dInu += [np.std(_tmp) / len(_tmp)**0.5]
        return rpnts, np.array(Inu), np.array(dInu)

    def radial_threshold(self, rotated_data, inc, nsigma=1.0, smooth=1.0,
                         think_positively=True, mask_value=0.0):
        """
        Calculates a radial profile of the peak flux density including the mean
        and the azimuthal scatter. The latter defines a threshold for clipping.

        Args:
            rotated_data (array): The data to mask, rotated such that the
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
        if nsigma is None:
            return rotated_data
        rvals = self.disk_coords(x0=0.0, y0=0.0, inc=inc, PA=90.0)[0]
        out = self.quick_peak_profile(inc, 90.0, np.max(rotated_data, axis=0))
        rpnts, avgTb, stdTb = out
        if smooth > 0.0:
            kernel = Gaussian1DKernel((smooth * self.bmaj) / self.dpix / 2.235)
            avgTb = convolve(avgTb, kernel, boundary='wrap')
            stdTb = convolve(stdTb, kernel, boundary='wrap')
        Inu_clip = interp1d(rpnts, avgTb - nsigma * stdTb,
                            bounds_error=False, fill_value=0.0)(rvals)
        return np.where(rotated_data >= Inu_clip, rotated_data, mask_value)

    def integrated_spectrum(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, r_max=None):
        """
        Returns the integrated spectrum over a specified region.

        Args:
            x0 (Optional[float]): Right Ascension offset in [arcsec].
            y0 (Optional[float]): Declination offset in [arcsec].
            inc (Optional[float]): Disk inclination in [deg].
            PA (Optional[float]): Disk position angle in [deg].
            r_max (Optional[float]): Radius to integrate out to in [arcsec].

        Returns:
            spectrum, uncertainty (array, array): Something about these.
        """
        rr = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA)[0]
        r_max = rr.max() if r_max is None else r_max
        nbeams = np.where(rr <= r_max, 1, 0).sum() / self.pix_per_beam
        spectrum = np.array([np.nansum(c[rr <= r_max]) for c in self.data])
        spectrum *= self.beams_per_pix
        uncertainty = np.sqrt(nbeams) * self.estimate_RMS()
        return spectrum, uncertainty

    def plot_spectrum(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, r_max=None):
        """
        Plot the integrated spectrum.

        Args:
            x0 (Optional[float]): Right Ascension offset in [arcsec].
            y0 (Optional[float]): Declination offset in [arcsec].
            inc (Optional[float]): Disk inclination in [deg].
            PA (Optional[float]): Disk position angle in [deg].
            r_max (Optional[float]): Radius to integrate out to in [arcsec].
        """
        x = self.velax.copy() / 1e3
        y, dy = self.integrated_spectrum(x0, y0, inc, PA, r_max)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        L = ax.step(x, y, where='mid')
        ax.errorbar(x, y, dy, fmt=' ', color=L[0].get_color(), zorder=-10)
        ax.set_xlabel("Velocity (km/s)")
        ax.set_ylabel("Integrated Flux (Jy)")
        ax.set_xlim(x[0], x[-1])
        ax2 = ax.twiny()
        ax2.set_xlim(0, x.size-1)
        ax2.set_xlabel("Channel Index")
        for i in range(10, x.size, 10):
            ax2.axvline(i, ls='--', lw=1.0, zorder=-15, color='0.8')

    def iterative_clip_emission_surface(self, r, z, Inu=None, v=None,
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
            Inu (optional[array]): Array of flux densities in [Jy/beam].
            v (optional[array]): Array of velocities in [m/s].
            nsigma (optional[float]): The number of standard deviations away
                from the running mean to consider a 'good fit'. A larger number
                is more conservative.
            niter (optional[int]): The number of iterations of clipping.
            window (optional[float]): The width of the window as a fraction of
                the beam major axis. This is only rough.
            min_sigma (optional[float]): Minimum standard deviation to use in
                the iterative clipping as a fraction of the beam major axis.
            return_mask (optional[bool]): Return the mask rather than the
                clipped arrays.

        Returns:
            r, z[, Inu, v] (array, array[, array, array])
        """
        to_return = observation._pack_arguments(r, z, Inu, v)
        idxs = np.argsort(to_return[0])
        to_return = [p[idxs] for p in to_return]
        window = int((window * self.bmaj) / np.diff(to_return[0]).mean())
        min_sigma = (min_sigma * self.bmaj)
        mask = observation.sigma_clip(to_return[1], nsigma=nsigma,
                                      niter=niter, window=window,
                                      min_sigma=min_sigma)
        return observation._return_arguments(to_return, mask, return_mask)

    def clip_rolling_scatter_threshold(self, r, z, Inu=None, v=None,
                                       inc=90.0, nbeams=0.25, window=None,
                                       return_mask=False):
        """
        Clip the data based on the value of the rolling scatter. As the
        vertical direction is related to the on-sky distance as,

        .. math::
            z = \frac{y}{sin(i)}

        then for some uncertainty in :math:`y`, ``npix``, we can calculate the
        associated scatter in :math:`z`.

        .. warning::
            For low inclinations this will translate to very large scatters.
            The default ``inc`` value will result in the minimum scatter in
            ``z``.

        Args:
            r (array): An array of radius values in [arcsec].
            z (array): An array of corresponding z values in [arcsec].
            Inu (optional[array]): Array of flux densities in [Jy/beam].
            v (optional[array]): Array of velocities in [m/s].
            inc (optional[float]): Inclination of disk in [degrees].
            nbeams (optional[float]): Threshold of the scatter as a fraction of
                the beam major axis.
            window (optional[int]): Window size for the rolling standard
                deviation. Defaults to quarter of the beam FWHM.
            return_mask (optional[bool]): Return the mask rather than the
                clipped arrays.

        Returns:
            r, z[, Inu, v] (array, array[, array, array])
        """
        to_return = observation._pack_arguments(r, z, Inu, v)
        if window is None:
            window = self.estimate_rolling_stats_window(to_return[0], 0.25)
        _, z_std = observation.rolling_stats(to_return[1], window=window)
        scatter_threshold = nbeams * self.bmaj / np.sin(np.radians(inc))
        for idx, scatter in enumerate(z_std):
            if scatter > scatter_threshold:
                break
        mask = to_return[0] < to_return[0][idx]
        return observation._return_arguments(to_return, mask, return_mask)

    def estimate_radial_range_counts(self, r, min_counts=8, window=8,
                                     bin_in_radius_kwargs=None):
        """
        Given some binning parameters, the number of points in each bin are
        counted. Starting from the bin with the most points in, the closest two
        bins where the counts fall below some threshold are given at the inner
        and outer bounds to clip the data.

        Args:
            r (array): Either array of the unbinned radial samples.
            min_counts (optional[int]): Minimum number of counts for each bin.
            window (optional[int]): Window size to use to calculate the rolling
                average of the bin counts.
            bin_in_radius_kwargs (optional[dict]): Kwargs to pass to
                ``bin_in_radius`` if ``bin_data=True``.

        Returns:
            r_min, r_max (float, float): Inner and outer radius of
        """

        kw = {} if bin_in_radius_kwargs is None else bin_in_radius_kwargs
        kw['statistic'] = 'count'
        r_bins, N_bins, _ = self.bin_in_radius(r, r, **kw)
        if window > 1:
            N_bins, _ = observation.rolling_stats(N_bins, window)
        idx = np.argmax(N_bins)

        if N_bins[idx] < min_counts:
            raise ValueError("No bin with {} counts.".format(min_counts))

        min_val = N_bins.max()
        for i in range(idx):
            min_val = min(N_bins[idx - i], min_val)
            if min_val < min_counts:
                break
        i += 1 if i == idx - 1 else 0
        r_min = r_bins[idx - i]

        min_val = N_bins.max()
        for i in range(r_bins.size - idx):
            min_val = min(N_bins[idx + i], min_val)
            if min_val < min_counts:
                break
        r_max = r_bins[idx + i]

        return r_min, r_max

    def clip_bin_counts(self, r, z, Inu=None, v=None, min_counts=8, window=8,
                        bin_in_radius_kwargs=None, return_mask=False):
        """
        Given some binning parameters, the number of points in each bin are
        counted. Starting from the bin with the most points in, the closest two
        bins where the counts fall below some threshold are given at the inner
        and outer bounds to clip the data.

        Args:
            r (array): Either array of the unbinned radial samples.
            min_counts (optional[int]): Minimum number of counts for each bin.
            window (optional[int]): Window size to use to calculate the rolling
                average of the bin counts.
            bin_in_radius_kwargs (optional[dict]): Kwargs to pass to
                ``bin_in_radius`` if ``bin_data=True``.
            return_mask (optional[bool]): Return the mask rather than the
                clipped arrays.

        Returns:
            r, z[, Inu, v] (array, array[, array, array])
        """
        to_return = observation._pack_arguments(r, z, Inu, v)
        r_min, r_max = self.estimate_radial_range_counts(r, min_counts, window,
                                                         bin_in_radius_kwargs)
        mask = np.logical_and(r >= r_min, r <= r_max)
        return observation._return_arguments(to_return, mask, return_mask)

    def estimate_radial_range(self, inc=0.0, PA=0.0, nsigma=5.0, nchan=None):
        """
        Estimate the radial range to consider in the emission surface. This is
        done by making a radial profile of the peak flux density, then only
        considering the radial range where the peak value is greater than
        ``nsigma * RMS`` where ``RMS`` is calculated based on the first and
        last ``nchan`` channels.

        Args:
            inc (optional[float]): Disk inclination in [degrees].
            PA (optional[float]): Disk position angle in [degrees].
            nsigma (optional[float]): The RMS factor used to clip the profile.
            nchan (optional[int]): The number of first and last channels to use
                to estimate the standard deviation of the spectrum.

        Returns:
            r_min, r_max (float, float): The inner and outer radius where the
                peak flux density is above the provided limit.
        """
        r, Inu, _ = self.quick_peak_profile(inc=inc, PA=PA)
        nchan = int(self.velax.size / 3) if nchan is None else int(nchan)
        if nchan > self.velax.size / 2 and self.verbose:
            print("WARNING: `nchan` larger than half the spectral axis.")
        Inu /= self.estimate_RMS(N=nchan)
        Inu = np.where(Inu >= nsigma, 1, 0)
        r_min = 0.0
        for i, F in enumerate(Inu):
            if F > 0:
                r_min = r[i]
                break
        r_max = self.xaxis.max()
        for i, F in enumerate(Inu[::-1]):
            if F > 0:
                r_max = r[r.size - 1 - i]
                break
        return 0.0 if r_min == r[0] else r_min, r_max

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
    def powerlaw(r, z0, q, r_cavity=0.0):
        """Standard power law profile."""
        return z0 * np.clip(r - r_cavity, a_min=0.0, a_max=None)**q

    @staticmethod
    def tapered_powerlaw(r, z0, q, r_taper=np.inf, q_taper=1.0, r_cavity=0.0):
        """Exponentially tapered power law profile."""
        rr = np.clip(r - r_cavity, a_min=0.0, a_max=None)
        f = observation.powerlaw(rr, z0, q)
        return f * np.exp(-(rr / r_taper)**q_taper)

    # -- PLOTTING FUNCTIONS -- #

    def plot_temperature(self, surface, side='both', reflect=False,
                         masked=True, ax=None, return_fig=False):
        """
        Plot the temperature structure from the surface.

        Args:
            TBD
        """

        # Generate plotting axes. If a previous axis has been provided, we
        # use the limits used for the most recent call of `plt.scatter` to set
        # the same `vmin` and `vmax` values for ease of comparison. We also
        # test to see if there's a second axis in the figure

        if ax is None:
            fig, ax = plt.subplots()
            min_T, max_T = None, None
            colorbar = True
        else:
            return_fig = False
            min_T, max_T = 1e10, -1e10
            for child in ax.get_children():
                try:
                    _min_T, _max_T = child.get_clim()
                    min_T = min(_min_T, min_T)
                    max_T = max(_max_T, max_T)
                except (AttributeError, TypeError):
                    continue
            colorbar = False

        # Plot each side separately to have different colors.

        r, z, Tb = np.empty(1), np.empty(1), np.empty(1)
        if side.lower() not in ['front', 'back', 'both']:
            raise ValueError(f"Unknown `side` value {side}.")
        if side.lower() in ['front', 'both']:
            r = np.concatenate([r, surface.r(side='front', masked=masked)])
            z = np.concatenate([z, surface.z(side='front', masked=masked)])
            _Tb = self.jybeam_to_Tb(surface.I(side='front', masked=masked))
            Tb = np.concatenate([Tb, _Tb])
        if side.lower() in ['back', 'both']:
            r = np.concatenate([r, surface.r(side='back', masked=masked)])
            _z = surface.z(side='back', reflect=reflect, masked=masked)
            z = np.concatenate([z, _z])
            _Tb = self.jybeam_to_Tb(surface.I(side='back', masked=masked))
            Tb = np.concatenate([Tb, _Tb])
        r, z, Tb = r[1:], z[1:], Tb[1:]
        min_T = np.nanmin(Tb) if min_T is None else min_T
        max_T = np.nanmax(Tb) if max_T is None else max_T

        # Three plots to include an outline without affecting the perceived
        # alpha of the points.

        ax.scatter(r, z, color='k', marker='o', lw=2.0)
        ax.scatter(r, z, color='w', marker='o', lw=1.0)
        ax.scatter(r, z, c=Tb, marker='o', lw=0.0, vmin=min_T,
                   vmax=max_T, alpha=0.2, cmap='RdYlBu_r')

        # Gentrification.

        ax.set_xlabel("Radius (arcsec)")
        ax.set_ylabel("Height (arcsec)")
        if colorbar:
            fig.set_size_inches(fig.get_figwidth() * 1.2,
                                fig.get_figheight(),
                                forward=True)
            im = ax.scatter(r, z * np.nan, c=Tb, marker='.', vmin=min_T,
                            vmax=max_T, cmap='RdYlBu_r')
            cb = plt.colorbar(im, ax=ax, pad=0.02)
            cb.set_label("T (K)", rotation=270, labelpad=13)

        # Returns.

        if return_fig:
            return fig

    def plot_channels(self, chans=None, velocities=None, return_fig=False):
        """
        Plot the channels within the channel range or velocity range.

        Args:
            chans
            return_fig
        """
        from matplotlib.ticker import MaxNLocator
        import matplotlib.pyplot as plt

        # Parse the channel and velocity ranges.

        if chans is not None and velocities is not None:
            raise ValueError("Only specify `chans` or `velocities`.")
        elif chans is None and velocities is None:
            chans = [0, self.velax.size - 1]
        elif velocities is not None:
            chans = [abs(self.velax - velocities[0]).argmin(),
                     abs(self.velax - velocities[1]).argmin()]
        assert chans[0] >= 0 and chans[1] <= self.velax.size - 1

        # Plot the channel map.

        velocities = self.velax.copy()[chans[0]:chans[1]+1]
        nrows = np.ceil(velocities.size / 5).astype(int)
        fig, axs = plt.subplots(ncols=5, nrows=nrows, figsize=(11, 2*nrows+1),
                                constrained_layout=True)
        for a, ax in enumerate(axs.flatten()):
            if a >= self.velax.size:
                continue
            ax.imshow(self.data[chans[0]+a], origin='lower',
                      extent=self.extent, vmax=0.75*np.nanmax(self.data),
                      vmin=0.0, cmap='binary_r')
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.grid(ls='--', lw=1.0, alpha=0.2)
            ax.text(0.05, 0.95, 'chan_idx = {:d}'.format(chans[0] + a),
                    fontsize=9, color='w', ha='left', va='top',
                    transform=ax.transAxes)
            ax.text(0.95, 0.95, '{:.2f} km/s'.format(velocities[a] / 1e3),
                    fontsize=9, color='w', ha='right', va='top',
                    transform=ax.transAxes)
            if ax != axs[-1, 0]:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            else:
                ax.set_xlabel('Offset (arcsec)')
                ax.set_ylabel('Offset (arcsec)')
        if axs.size != velocities.size:
            for ax in axs.flatten()[-(axs.size - velocities.size):]:
                ax.axis('off')

        if return_fig:
            return fig

    def plot_isovelocities(self, surface, mstar, vlsr, dist, side='both',
                           reflect=True, smooth=None, return_fig=False):
        """
        Plot the isovelocity contours for the given emission surface.

        Args:
            surface
            mstar
            vlsr
            dist
            side
            reflect
            return_fig
        """
        from matplotlib.ticker import MaxNLocator
        from scipy.interpolate import interp1d
        import matplotlib.pyplot as plt

        if side not in ['front', 'back', 'both']:
            raise ValueError(f"Unknown `side` value {side}.")

        velocities = self.velax[surface.chans.min():surface.chans.max()+1]
        nrows = np.ceil(velocities.size / 5).astype(int)
        fig, axs = plt.subplots(ncols=5, nrows=nrows, figsize=(11, 2*nrows+1),
                                constrained_layout=True)

        # Define the function for the front surface. As `reflect=True` will
        # just reflect the front side about the midplane, we will always
        # calculate this just in case.

        r, z, _ = surface.rolling_surface(side='front')
        z[np.logical_and(r >= surface.r_min, r <= surface.r_min)] = np.nan
        z = surface.convolve(z, smooth) if smooth is not None else z
        z_f = interp1d(r, z, bounds_error=False, fill_value=np.nan)

        # Define the function for the back surface.

        if reflect:
            z_b = interp1d(r, -z, bounds_error=False, fill_value=np.nan)
        else:
            r, z, _ = surface.rolling_surface(side='back')
            z[np.logical_and(r >= surface.r_min, r <= surface.r_min)] = np.nan
            z = surface.convolve(z, smooth) if smooth is not None else z
            z_b = interp1d(r, z, bounds_error=False, fill_value=np.nan)

        # Calculate the projected velocity maps for both sides of the disk.

        v_f = self.keplerian(inc=surface.inc, PA=surface.PA, mstar=mstar,
                             dist=dist, x0=surface.x0, y0=surface.y0,
                             vlsr=vlsr, z_func=z_f)

        v_b = self.keplerian(inc=surface.inc, PA=surface.PA, mstar=mstar,
                             dist=dist, x0=surface.x0, y0=surface.y0,
                             vlsr=vlsr, z_func=z_b)

        # Plot the contours.

        for vv, ax in zip(velocities, axs.flatten()):

            channel = self.data[abs(self.velax - vv).argmin()]
            ax.imshow(channel, origin='lower', extent=self.extent,
                      vmax=0.75*np.nanmax(self.data), vmin=0.0,
                      cmap='binary_r')

            if side in ['front', 'both']:
                ax.contour(self.xaxis, self.yaxis, v_f, [vv], colors='b')
            if side in ['back', 'both']:
                ax.contour(self.xaxis, self.yaxis, v_b, [vv], colors='r')

            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.grid(ls='--', lw=1.0, alpha=0.2)

            if ax != axs[-1, 0]:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            else:
                ax.set_xlabel('Offset (arcsec)')
                ax.set_ylabel('Offset (arcsec)')

        if axs.size != velocities.size:
            for ax in axs.flatten()[-(axs.size - velocities.size):]:
                ax.axis('off')

        if return_fig:
            return fig

    def plot_peaks(self, surface, side='both', return_fig=False):
        """
        Plot the peak locations on channel maps.

        Args:
            surface (surface instance): The extracted surface returned from
                ``get_emission_surface``.
            side (Optional[str]): Side to plot. Must be ``'front'``, ``'back'``
                or ``'both'``. Defaults to ``'both'``.
            return_fig (Optional[bool]): Whether to return the Matplotlib
                figure. Defaults to ``True``.
        """
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        velocities = self.velax[surface.chans.min():surface.chans.max()+1]
        nrows = np.ceil(velocities.size / 5).astype(int)
        fig, axs = plt.subplots(ncols=5, nrows=nrows, figsize=(11, 2*nrows+1),
                                constrained_layout=True)

        velax = self.velax[surface.chans.min():surface.chans.max()+1]
        data = self.data_aligned_rotated[surface.data_aligned_rotated_key]

        for vv, ax in zip(velocities, axs.flatten()):

            channel = data[abs(velax - vv).argmin()]

            ax.imshow(channel, origin='lower', extent=self.extent,
                      cmap='binary_r', vmin=0.0, vmax=0.75*data.max())

            if side.lower() in ['back', 'both']:
                toplot = surface.v(side='back') == vv
                ax.scatter(surface.x(side='back')[toplot],
                           surface.y(side='back', edge='far')[toplot],
                           lw=0.0, color='r', marker='.')
                ax.scatter(surface.x(side='back')[toplot],
                           surface.y(side='back', edge='near')[toplot],
                           lw=0.0, color='r', marker='.')

            if side.lower() in ['front', 'both']:
                toplot = surface.v(side='front') == vv
                ax.scatter(surface.x(side='front')[toplot],
                           surface.y(side='front', edge='far')[toplot],
                           lw=0.0, color='b', marker='.')
                ax.scatter(surface.x(side='front')[toplot],
                           surface.y(side='front', edge='near')[toplot],
                           lw=0.0, color='b', marker='.')

            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.grid(ls='--', lw=1.0, alpha=0.2)

            if ax != axs[-1, 0]:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            else:
                ax.set_xlabel('Offset (arcsec)')
                ax.set_ylabel('Offset (arcsec)')

        if axs.size != velocities.size:
            for ax in axs.flatten()[-(axs.size - velocities.size):]:
                ax.axis('off')

        if return_fig:
            return fig
