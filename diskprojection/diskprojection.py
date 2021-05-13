"""
A wrapper class to allow a user to extract an emission surface from a datacube
of molecular line emission. This uses the method presented in Pinte et al.
(2018).
"""
from astropy.convolution import convolve, Gaussian1DKernel
from scipy.interpolate import interp1d
from .detect_peaks import detect_peaks
from gofish import imagecube
import numpy as np
import time


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
                             r_min=None, r_max=None, smooth=0.5,
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
            r, z, Fnu, v (arrays): Arrays of radius, height, flux density and
                velocity.
        """

        # Remove bad inclination:
        if inc == 0.0:
            raise ValueError("Cannot infer height with face on disk.")
        if self.verbose and abs(inc) < 10.0:
            print("WARNING: Inferences with such face on disk are poor.")

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
                if not r_min <= abs(self.xaxis[x_idx]) <= r_max:
                    continue
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
                peaks += [[r, z, Fnu, self.velax[chans[0]:chans[1]+1][c_idx]]]
        peaks = np.squeeze(peaks).T
        peaks = peaks[:, np.isfinite(peaks[2])]

        # Sort the values in increasing radius.
        if return_sorted:
            idxs = np.argsort(peaks[0])
            peaks = [p[idxs] for p in peaks]
        return peaks

    def quick_peak_profile(self, inc, PA, data=None):
        """
        Returns a quick and dirty radial profile of the peak flux density. This
        function does not consider any flared emission surfaces, offset  and
        only takes the maximum value along the spectral axis.

        Args:
            inc (float): Disk inclination in [degrees].
            PA (float): Disk position angle in [degrees].
            data (optional): Data to make a profile of. If no data is provided,
                take the maximum of ``self.data`` along the spectral axis.

        Returns:
            r, Fnu, dFnu (array, array, array): Arrays of the peak flux
                density, ``Fnu`` at radial positions ``r``. ``dFnu`` is given
                by the standard error on the mean.
        """
        data = np.nanmax(self.data.copy(), axis=0) if data is None else data
        if data.ndim != 2:
            raise ValueError("`data` must be a 2D array.")
        data = data.flatten()
        rbins, rpnts = self.radial_sampling()
        rvals = self.disk_coords(x0=0.0, y0=0.0, inc=inc, PA=PA)[0]
        ridxs = np.digitize(rvals.flatten(), rbins)
        Fnu, dFnu = [], []
        for r in range(1, rbins.size):
            _tmp = data[ridxs == r]
            _tmp = _tmp[np.isfinite(_tmp)]
            Fnu += [np.mean(_tmp)]
            dFnu += [np.std(_tmp) / len(_tmp)**0.5]
        return rpnts, np.array(Fnu), np.array(dFnu)

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
        rvals = self.disk_coords(x0=0.0, y0=0.0, inc=inc, PA=0.0)[0]
        out = self.quick_peak_profile(inc, 0.0, np.max(rotated_data, axis=0))
        rpnts, avgTb, stdTb = out
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
            return_mask (optional[bool]): Return the mask rather than the
                clipped arrays.

        Returns:
            r, z[, Fnu, v] (array, array[, array, array])
        """
        to_return = disk_observation._pack_arguments(r, z, Fnu, v)
        idxs = np.argsort(to_return[0])
        to_return = [p[idxs] for p in to_return]
        window = int((window * self.bmaj) / np.diff(to_return[0]).mean())
        min_sigma = (min_sigma * self.bmaj)
        mask = disk_observation.sigma_clip(to_return[1], nsigma=nsigma,
                                           niter=niter, window=window,
                                           min_sigma=min_sigma)
        return disk_observation._return_arguments(to_return, mask, return_mask)

    def clip_emission_height(self, r, z, Fnu=None, v=None, z_min=None,
                             z_max=None, return_mask=False):
        """
        Clip the data based on absolute values on the emission height.

        Args:
            r (array): An array of radius values in [arcsec].
            z (array): An array of corresponding z values in [arcsec].
            Fnu (optional[array]): Array of flux densities in [Jy/beam].
            v (optional[array]): Array of velocities in [m/s].
            z_min (optional[float]): Minimum value of z to include in [arcsec].
            z_max (optional[float]): Maximum value of z to include in [arcsec].
            return_mask (optional[bool]): Return the mask rather than the
                clipped arrays.

        Returns:
            r, z[, Fnu, v] (array, array[, array, array])
        """
        to_return = disk_observation._pack_arguments(r, z, Fnu, v)
        z_min = z.min() if z_min is None else z_min
        z_max = z.max() if z_max is None else z_max
        if not z_min < z_max:
            raise ValueError("`z_min` must be smaller than `z_max`.")
        mask = np.logical_and(z >= z_min, z <= z_max)
        return disk_observation._return_arguments(to_return, mask, return_mask)

    def clip_aspect_ratio(self, r, z, Fnu=None, v=None, zr_min=None,
                          zr_max=None, return_mask=False):
        """
        Clip the data based on the aspect ratio of the emission, :math:`z/r`.

        Args:
            r (array): An array of radius values in [arcsec].
            z (array): An array of corresponding z values in [arcsec].
            Fnu (optional[array]): Array of flux densities in [Jy/beam].
            v (optional[array]): Array of velocities in [m/s].
            zr_min (optional[float]): Minimum aspect ratio to include.
            zr_max (optional[float]): Maximum aspect ratio to include.
            return_mask (optional[bool]): Return the mask rather than the
                clipped arrays.

        Returns:
            r, z[, Fnu, v] (array, array[, array, array])
        """
        to_return = disk_observation._pack_arguments(r, z, Fnu, v)
        zr_min = -1e2 if zr_min is None else zr_min
        zr_max = 1e2 if zr_max is None else zr_max
        if not zr_min < zr_max:
            raise ValueError("`zr_min` must be smaller than `zr_max`.")
        mask = np.logical_and(z/r >= zr_min, z/r <= zr_max)
        return disk_observation._return_arguments(to_return, mask, return_mask)

    def clip_flux_density(self, r, z, Fnu, v=None, Fnu_min=None, Fnu_max=None,
                          return_mask=False):
        """
        Clip the data based on the flux density, :math:`F_{\nu}`.

        Args:
            r (array): An array of radius values in [arcsec].
            z (array): An array of corresponding z values in [arcsec].
            Fnu (optional[array]): Array of flux densities in [Jy/beam].
            v (optional[array]): Array of velocities in [m/s].
            Fnu_min (optional[float]): Minimum flux density to include in
                [Jy/beam].
            Fnu_max (optional[float]): Maximum flux density to include in
                [Jy/beam].
            return_mask (optional[bool]): Return the mask rather than the
                clipped arrays.

        Returns:
            r, z[, Fnu, v] (array, array[, array, array])
        """
        to_return = disk_observation._pack_arguments(r, z, Fnu, v)
        Fnu_min = Fnu.min() if Fnu_min is None else Fnu_min
        Fnu_max = Fnu.max() if Fnu_max is None else Fnu_max
        if not Fnu_min < Fnu_max:
            raise ValueError("`Fnu_min` must be smaller than `Fnu_max`.")
        mask = np.logical_and(Fnu >= Fnu_min, Fnu <= Fnu_max)
        return disk_observation._return_arguments(to_return, mask, return_mask)

    def clip_rolling_scatter_threshold(self, r, z, Fnu=None, v=None,
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
            Fnu (optional[array]): Array of flux densities in [Jy/beam].
            v (optional[array]): Array of velocities in [m/s].
            inc (optional[float]): Inclination of disk in [degrees].
            nbeams (optional[float]): Threshold of the scatter as a fraction of
                the beam major axis.
            window (optional[int]): Window size for the rolling standard
                deviation. Defaults to quarter of the beam FWHM.
            return_mask (optional[bool]): Return the mask rather than the
                clipped arrays.

        Returns:
            r, z[, Fnu, v] (array, array[, array, array])
        """
        to_return = disk_observation._pack_arguments(r, z, Fnu, v)
        if window is None:
            window = self.estimate_rolling_stats_window(to_return[0], 0.25)
        _, z_std = disk_observation.rolling_stats(to_return[1], window=window)
        scatter_threshold = nbeams * self.bmaj / np.sin(np.radians(inc))
        for idx, scatter in enumerate(z_std):
            if scatter > scatter_threshold:
                break
        mask = to_return[0] < to_return[0][idx]
        return disk_observation._return_arguments(to_return, mask, return_mask)

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
            N_bins, _ = disk_observation.rolling_stats(N_bins, window)
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

    def clip_bin_counts(self, r, z, Fnu=None, v=None, min_counts=8, window=8,
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
            r, z[, Fnu, v] (array, array[, array, array])
        """
        to_return = disk_observation._pack_arguments(r, z, Fnu, v)
        r_min, r_max = self.estimate_radial_range_counts(r, min_counts, window,
                                                         bin_in_radius_kwargs)
        mask = np.logical_and(r >= r_min, r <= r_max)
        return disk_observation._return_arguments(to_return, mask, return_mask)

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
        r, Fnu, _ = self.quick_peak_profile(inc=inc, PA=PA)
        nchan = int(self.velax.size / 3) if nchan is None else int(nchan)
        if nchan > self.velax.size / 2 and self.verbose:
            print("WARNING: `nchan` larger than half the spectral axis.")
        Fnu /= self.estimate_RMS(N=nchan)
        Fnu = np.where(Fnu >= nsigma, 1, 0)
        r_min = 0.0
        for i, F in enumerate(Fnu):
            if F > 0:
                r_min = r[i]
                break
        r_max = self.xaxis.max()
        for i, F in enumerate(Fnu[::-1]):
            if F > 0:
                r_max = r[r.size - 1 - i]
                break
        return 0.0 if r_min == r[0] else r_min, r_max

    def estimate_channel_range(self, average='mean', nsigma=5.0, nchan=None,
                               minimum_mask_size=5):
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
            minimum_mask_size (optional[int]): If the mask size is less than
                 this value, return ``None`` and print a warning.

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
        if sum(mask) < minimum_mask_size:
            if self.verbose:
                print("WARNING: Mask smaller than `minimum_mask_size`.")
                print("\t Returning `chans=None` instead.")
            return None
        return [self.channels[mask][0], self.channels[mask][-1]]

    def estimate_rolling_stats_window(self, r, nbeams=1.0, check_ordered=True):
        """
        Estimate the window size for ``rolling_stats`` to match the requested
        number of beam FWHMs.

        Args:
            r (list): A list of monotonically increasing radial points.
            nbeams (optional[float]): Desired window size as a fraction of the
                beam major FWHM.
            check_ordered (optional[bool]): Check that ``r`` is monotonically
                increasing. If not, raise a ``ValueError``.

        Returns:
            window_size (int): The size of the window that best reproduces the
                requested scale.
        """
        dr = np.diff(r)
        if check_ordered and any(dr < 0):
            raise ValueError("Values in `r` are not monotonically increasing.")
        dr = np.where(dr == 0.0, 1e-10, dr)
        return np.median(nbeams * self.bmaj / dr).astype('int')

    def bin_in_radius(self, r, x, rbins=None, rvals=None, statistic='mean',
                      uncertainty='std'):
        """
        Radially bin the ``x`` data. The error can either be the standard
        deviation in the bin (``uncertainty='std'``), the 16th to 84th
        percentiles (``uncertainty='percentiles'``) or the standard erron on
        the mean (``uncertainty='standard'``).

        Args:
            r (list): A list of the radial points in [arcsec].
            x (list): A list of the poinst to bin.
            rbins (optional[list]): A list of the bin edges to use.
            rpnts (optional[list]): A list of the bin centers to use. The edge
                bins are calculated assuming equal size bins.
            statistic (optional[str]): Statistic to use for
                ``scipy.stats.binned_statistic``. If ``'mean'`` or ``'median'``
                are provided, will default to the Numpy functions which can
                handle NaNs.
            uncertainty (optional[str]): Type of uncertainty to use for the
                binned values: ``'std'``, ``'percentiles'`` or ``'standard'``.

        Returns:
            rvals, z_avg, z_err (array, array, array: Arrays of the bin
                centres, bin average and bin uncertainties.
        """
        from scipy.stats import binned_statistic
        rbins, rvals = self.radial_sampling(rbins=rbins, rvals=rvals)

        if statistic.lower() == 'mean':
            statistic = np.nanmean
        elif statistic.lower() == 'median':
            statistic = np.nanmedian
        z_avg = binned_statistic(r, x, bins=rbins, statistic=statistic)[0]

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

        elif uncertainty.lower() == 'standard':
            z_err = binned_statistic(r, x, bins=rbins, statistic=np.nanstd)[0]
            npnts = binned_statistic(r, x, bins=rbins, statistic='count')[0]
            z_err /= np.sqrt(npnts)

        else:
            warning = "Unknown `uncertainty` value, {}."
            raise ValueError(warning.format(uncertainty))

        return rvals, z_avg, z_err

    def fit_emission_surface(self, r, z, dz=None, tapered_powerlaw=True,
                             include_cavity=False, curve_fit_kwargs=None):
        """
        Fit the inferred emission surface with a tapered power law of the form

        .. math::
            z(r) = z_0 \, \left( \frac{r}{1^{\prime\prime}} \right)^{\psi}
            \times \exp \left( -\left[ \frac{r}{r_{\rm taper}}
            \right]^{\psi_{\rm taper}} \right)

        where a single power law profile is recovered when
        :math:`r_{\rm taper} \rightarrow \infty`, and can be forced using the
        ``tapered_powerlaw=False`` argument.

        We additionally allow for an inner cavity, :math:`r_{\rm cavity}`,
        inside which all emission heights are set to zero, and the radial range
        is shifted such that :math:`r^{\prime} = r - r_{\rm cavity}`. This can
        be toggled with the ``include_cavity`` argument.

        The fitting is performed with ``scipy.optimize.curve_fit`` where the
        returned uncertainties are the square root of the diagnal components of
        the covariance maxtrix returned by ``curve_fit``.

        Args:
            r (array): An array of radius values in [arcsec].
            z (array): An array of corresponding z values in [arcsec].
            dz (optional[array]): An array of uncertainties for the z values in
                [arcsec]. If these are absolute uncertainties, make sure to
                include the ``absolute_sigma=True`` argument in the kwargs.
            tapered_powerlaw (optional[bool]): If ``True``, fit the tapered
                power law profile rather than a single power law function.
            include_cavity (optional[bool]): If ``True``, include a cavity in
                the functional form, inside of which all heights are set to 0.
            curve_fit_kwargs (optional[dict]): Keyword arguments to pass to
                ``scipy.optimize.curve_fit``.

        Returns:
            popt, copy (array, array): Best-fit values and associated
                uncertainties for the fits.
        """
        from scipy.optimize import curve_fit
        kw = {} if curve_fit_kwargs is None else curve_fit_kwargs
        kw['maxfev'] = kw.pop('maxfev', 100000)
        kw['sigma'] = kw.pop('sigma', dz)
        kw['p0'] = kw.pop('p0', [0.3, 1.0, 1.0, 1.0, 0.05])
        if not include_cavity and len(kw['p0']) % 2:
            kw['p0'] = kw['p0'][:-1]
        if not tapered_powerlaw:
            kw['p0'] = kw['p0'][:2] + kw['p0'][4:]
        try:
            func = disk_observation.tapered_powerlaw
            popt, copt = curve_fit(func, r, z, **kw)
            copt = np.diag(copt)**0.5
        except RuntimeError:
            if self.verbose:
                print("WARNING: Failed to fit the data. Returning `p0`.")
            popt = kw['p0']
            copt = [np.nan for _ in popt]
        return popt, copt

    def fit_emission_surface_MCMC(self, r, z, dz=None, tapered_powerlaw=True,
                                  include_cavity=False, p0=None, nwalkers=64,
                                  nburnin=1000, nsteps=500, scatter=1e-3,
                                  priors=None, returns=None, plots=None,
                                  curve_fit_kwargs=None, niter=1):
        """
        Fit the inferred emission surface with a tapered power law of the form

        .. math::
            z(r) = z_0 \, \left( \frac{r}{1^{\prime\prime}} \right)^{\psi}
            \times \exp \left( -\left[ \frac{r}{r_{\rm taper}}
            \right]^{\psi_{\rm taper}} \right)

        where a single power law profile is recovered when
        :math:`r_{\rm taper} \rightarrow \infty`, and can be forced using the
        ``tapered_powerlaw=False`` argument.

        We additionally allow for an inner cavity, :math:`r_{\rm cavity}`,
        inside which all emission heights are set to zero, and the radial range
        is shifted such that :math:`r^{\prime} = r - r_{\rm cavity}`. This can
        be toggled with the ``include_cavity`` argument.

        The fitting (or more acurately the estimation of the posterior
        distributions) is performed with ``emcee``. If starting positions are
        not provided, will use ``fit_emission_surface`` to estimate starting
        positions.

        The priors are provided by a dictionary where the keys are the relevant
        argument names. Each param is described by two values and the type of
        prior. For a flat prior, ``priors['name']=[min_val, max_val, 'flat']``,
        while for a Gaussian prior,
        ``priors['name']=[mean_val, std_val, 'gaussian']``.

        Args:
            r (array): An array of radius values in [arcsec].
            z (array): An array of corresponding z values in [arcsec].
            dz (optional[array]): An array of uncertainties for the z values in
                [arcsec]. If nothing is given, assume an uncertainty equal to
                the pixel scale of the attached cube.
            tapered_powerlaw (optional[bool]): Whether to include a tapered
                component to the powerlaw.
            include_cavity (optional[bool]): Where to include an inner cavity.
            p0 (optional[list]): Starting guesses for the fit. If nothing is
                provided, will try to guess from the results of
                ``fit_emission_surface``.
            nwalkers (optional[int]): Number of walkers for the MCMC.
            nburnin (optional[int]): Number of steps to take to burn in.
            nsteps (optional[int]): Number of steps used to sample the PDF.
            scatter (optional[float]): Relative scatter used to randomize the
                starting positions of the walkers.
            priors (optional[dict]): A dictionary of priors to use for the
                fitting.
            returns (optional[list]): A list of properties to return. Can
                include: ``'samples'``, for the array of PDF samples (default);
                ``'percentiles'``, for the 16th, 50th and 84th percentiles of
                the PDF; ``'lnprob'`` for values of the log-probablity for each
                of the PDF samples; 'median' for the median value of the PDFs
                and ``'walkers'`` for the walkers.
            plots (optional[list]): A list of plots to make, including
                ``'corner'`` for the standard corner plot, or ``'walkers'`` for
                the trace of the walkers.
            curve_fit_kwargs (optional[dict]): Kwargs to pass to
                ``scipy.optimize.curve_fit`` if the ``p0`` values are estimated
                through ``fit_emision_surface``.

        Returns:
            Dependent on the ``returns`` argument.
        """
        import emcee

        # Remove any NaNs.
        nan_mask = np.isfinite(r) & np.isfinite(z)
        if dz is not None:
            nan_mask = nan_mask * np.isfinite(dz)
            r, z, dz = r[nan_mask], z[nan_mask], dz[nan_mask]
        else:
            r, z = r[nan_mask], z[nan_mask]

        # Define the initial guesses.
        if p0 is None:
            p0 = [0.3, 1.0, 1.0, 1.0, 0.05]
            if not include_cavity and len(p0) % 2:
                p0 = p0[:-1]
            if not tapered_powerlaw:
                p0 = p0[:2] + p0[4:]
            niter += 1

        nwalkers = max(nwalkers, 2 * len(p0))

        # Define the labels.
        labels = ['z0', 'psi']
        if tapered_powerlaw:
            if len(p0) < 4:
                raise ValueError("`p0` too short for a tapered powerlaw.")
            labels += ['r_taper', 'q_taper']
        if include_cavity:
            if not len(p0) % 2:
                raise ValueError("Even number of `p0` values; no `r_cavity`.")
            if p0[-1] <= 0.0:
                p0[-1] = 0.5
            labels += ['r_cavity']
        if self.verbose:
            print("Assuming p0 = {}.".format(labels))
            time.sleep(0.5)

        # Set the priors for the MCMC.
        priors = {} if priors is None else priors
        priors['z0'] = priors.pop('z0', [0.0, 5.0, 'flat'])
        priors['psi'] = priors.pop('psi', [0.0, 5.0, 'flat'])
        priors['r_taper'] = priors.pop('r_taper', [0.0, 15.0, 'flat'])
        priors['q_taper'] = priors.pop('q_taper', [0.0, 5.0, 'flat'])
        priors['r_cavity'] = priors.pop('r_cavity', [0.0, 15.0, 'flat'])

        # Check the uncertainties (/weights).
        dz = np.ones(z.size) * self.dpix if dz is None else dz

        # Set the starting positions for the walkers.

        for _ in range(niter):
            p0 = disk_observation._random_p0(p0, scatter, nwalkers)
            sampler = emcee.EnsembleSampler(nwalkers, p0.shape[1],
                                            self._ln_probability,
                                            args=(r, z, dz, labels, priors))
            sampler.run_mcmc(p0, nburnin + nsteps, progress=True)
            samples = sampler.chain[:, -int(nsteps):]
            samples = samples.reshape(-1, samples.shape[-1])
            p0 = np.median(samples, axis=0)
        walkers = sampler.chain.T

        # Diagnostic plots.
        plots = [] if plots is None else plots
        if 'walkers' in plots:
            disk_observation._plot_walkers(walkers, labels, nburnin)
        if 'corner' in plots:
            disk_observation._plot_corner(samples, labels)

        # Generate the output.
        to_return = []
        for r in ['samples'] if returns is None else np.atleast_1d(returns):
            if r == 'walkers':
                to_return += [walkers]
            if r == 'samples':
                to_return += [samples]
            if r == 'lnprob':
                to_return += [sampler.lnprobability[nburnin:]]
            if r == 'percentiles':
                to_return += [np.percentile(samples, [16, 50, 84], axis=0)]
            if r == 'median':
                to_return += [np.median(samples, axis=0)]
        return to_return if len(to_return) > 1 else to_return[0]

    def _ln_probability(self, theta, r, z, dz, labels, priors):
        """
        Log-probabiliy function for the emission surface fitting.

        Args:
            theta (array):
            r (array):
            z (array):
            dz (array):
            labels (list): List of label names.
            priors (dict): A dictionary of prior definitions. See ``_ln_prior``
                for more information on the format.

        Returns:
            lnx2 (float): Log-probaility value.
        """

        lnp = 0.0
        for label, t in zip(labels, theta):
            lnp += disk_observation._ln_prior(priors[label], t)
        if not np.isfinite(lnp):
            return lnp

        z0, q = theta[0], theta[1]
        try:
            r_taper = theta[labels.index('r_taper')]
            q_taper = theta[labels.index('q_taper')]
        except ValueError:
            r_taper = np.inf
            q_taper = 1.0
        try:
            r_cavity = theta[labels.index('r_cavity')]
        except ValueError:
            r_cavity = 0.0

        model = disk_observation.tapered_powerlaw(r, z0, q, r_taper,
                                                  q_taper, r_cavity)

        lnx2 = -0.5 * np.sum(np.power((z - model) / dz, 2)) + lnp
        return lnx2 if np.isfinite(lnx2) else -np.inf

    # -- Static Methods -- #

    @staticmethod
    def _plot_corner(samples, labels):
        """Make a corner plot."""
        try:
            import corner
        except ImportError:
            print("Must install `corner` to make corner plots.")
        corner.corner(samples, labels=labels, show_titles=True)

    @staticmethod
    def _plot_walkers(walkers, labels, nburnin):
        import matplotlib.pyplot as plt
        for param, label in zip(walkers, labels):
            fig, ax = plt.subplots()
            for walker in param.T:
                ax.plot(walker, alpha=0.1)
            ax.axvline(nburnin)
            ax.set_ylabel(label)
            ax.set_xlabel('Steps')

    @staticmethod
    def _random_p0(p0, scatter, nwalkers):
        """Get the starting positions."""
        p0 = np.squeeze(p0)
        dp0 = np.random.randn(nwalkers * len(p0)).reshape(nwalkers, len(p0))
        dp0 = np.where(p0 == 0.0, 1.0, p0)[None, :] * (1.0 + scatter * dp0)
        return np.where(p0[None, :] == 0.0, dp0 - 1.0, dp0)

    @staticmethod
    def _ln_prior(prior, theta):
        """
        Log-prior function. This is provided by two values and the type of
        prior. For a flat prior, ``prior=[min_val, max_val, 'flat']``, while
        for a Gaussianprior, ``prior=[mean_val, std_val, 'gaussian']``.

        Args:
            prior (tuple): Prior description.
            theta (float): Variable value.

        Returns:
            lnp (float): Log-prior probablity value.
        """
        if prior[2] == 'flat':
            if not prior[0] <= theta <= prior[1]:
                return -np.inf
            return 0.0
        lnp = -0.5 * ((theta - prior[0]) / prior[1])**2
        return lnp - np.log(prior[1] * np.sqrt(2.0 * np.pi))

    @staticmethod
    def _pack_arguments(r, z, Fnu=None, v=None):
        """Return the packed (r, z[, Fnu, v]) tuple."""
        to_return = [r, z]
        if Fnu is not None:
            to_return += [Fnu]
        if v is not None:
            to_return += [v]
        return to_return

    @staticmethod
    def _return_arguments(to_return, mask, return_mask):
        """Unpacks the arugments to return."""
        if return_mask:
            return mask
        return [p[mask] for p in to_return]

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
    def rolling_stats(x, window=51, average='mean'):
        """Rolling average and standard deviation."""
        w = window if window % 2 else window + 1
        edge = int((w - 1) / 2)
        xx = np.insert(x, 0, x[0] * np.ones(edge))
        xx = np.insert(xx, -1, x[-1] * np.ones(edge))
        if average.lower() == 'mean':
            avg = np.mean
        elif average.lower() == 'median':
            avg = np.median
        else:
            raise ValueError("Unknown `average` value: '{}''.".format(average))
        avg = np.squeeze([avg(xx[i+edge:i+w+edge]) for i in range(x.size)])
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

    @staticmethod
    def powerlaw(r, z0, q, r_cavity=0.0):
        """Standard power law profile."""
        return z0 * np.clip(r - r_cavity, a_min=0.0, a_max=None)**q

    @staticmethod
    def tapered_powerlaw(r, z0, q, r_taper=np.inf, q_taper=1.0, r_cavity=0.0):
        """Tapered power law profile."""
        rr = np.clip(r - r_cavity, a_min=0.0, a_max=None)
        f = disk_observation.powerlaw(rr, z0, q)
        return f * np.exp(-(rr / r_taper)**q_taper)
