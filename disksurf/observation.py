from .detect_peaks import detect_peaks
from astropy.convolution import convolve, Gaussian2DKernel
import matplotlib.pyplot as plt
from .surface import surface
from gofish import imagecube
import numpy as np


class observation(imagecube):
    """
    Wrapper of a GoFish imagecube class containing the emission surface
    extraction methods.

    Args:
        path (str): Relative path to the FITS cube.
        FOV (optional[float]): Clip the image cube down to a specific
            field-of-view spanning a range ``FOV``, where ``FOV`` is in
            [arcsec].
        velocity_range (optional[tuple]): A tuple of the minimum and maximum
            velocity in [m/s] to cut the cube down to.
    """

    def __init__(self, path, FOV=None, velocity_range=None):
        super().__init__(path=path, FOV=FOV, velocity_range=velocity_range)
        self.data_aligned_rotated = {}
        self.mask_keplerian = {}

    def get_emission_surface(self, inc, PA, x0=0.0, y0=0.0, vlsr=0.0,
                             chans=None, r_min=None, r_max=None, smooth=None,
                             nsigma=None, min_SNR=5, detect_peaks_kwargs=None,
                             get_keplerian_mask_kwargs=None,
                             force_opposite_sides=True):
        """
        Implementation of the method described in Pinte et al. (2018). There
        are several pre-processing options to help with the peak detection.

        Args:
            inc (float): Disk inclination in [degrees].
            PA (float): Disk position angle in [degrees].
            x0 (optional[float]): Disk offset along the x-axis in [arcsec].
            y0 (optional[float]): Disk offset along the y-axis in [arcsec].
            vlsr (optional[float]): Systemic velocity in [m/s].
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
                ``detect_peaks``. If any values are duplicated from those
                required for ``get_emission_surface``, they will be
                overwritten.
            force_opposite_sides (optional[bool]): Whether to assert that all
                pairs of peaks have one on either side of the major axis. By
                default this is ``True`` which is a more conservative approach
                but results in a lower sensitivity in the outer disk.

        Returns:
            A ``disksurf.surface`` instance containing the extracted emission
            surface.
        """

        # Grab the cut down and masked data.

        r_min = r_min or 0.0
        r_max = r_max or self.xaxis.max()
        if r_min >= r_max:
            raise ValueError("`r_min` must be less than `r_max`.")

        chans, data = self.get_aligned_rotated_data(inc=inc,
                                                    PA=PA,
                                                    x0=x0,
                                                    y0=y0,
                                                    chans=chans,
                                                    r_min=r_min,
                                                    r_max=r_max)

        # Calculate a Keplerian mask.
        # TODO: Check that the PA is defined correctly.

        if get_keplerian_mask_kwargs is not None:
            duplicates = ['x0', 'y0', 'inc', 'PA', 'r_min', 'r_max']
            if any([f in get_keplerian_mask_kwargs for f in duplicates]):
                msg = "Duplicate argument found in get_keplerian_mask_kwargs."
                print("WARNING: " + msg + "Overwriting parameters.")
            get_keplerian_mask_kwargs['x0'] = 0.0
            get_keplerian_mask_kwargs['y0'] = 0.0
            get_keplerian_mask_kwargs['inc'] = inc
            get_keplerian_mask_kwargs['PA'] = 90.0
            get_keplerian_mask_kwargs['r_min'] = r_min
            get_keplerian_mask_kwargs['r_max'] = r_max
            mask = self.get_keplerian_mask(**get_keplerian_mask_kwargs)
        else:
            mask = np.ones(data.shape).astype('bool')
        data = np.where(mask, data, 0.0)

        # Define the smoothing kernel.

        if smooth or 0.0 > 0.0:
            kernel = np.hanning(2.0 * smooth * self.bmaj / self.dpix)
            kernel /= np.sum(kernel)
        else:
            kernel = None

        # Find all the peaks.

        if self.verbose:
            print("Detecting peaks...")
        _surf = self._detect_peaks(data=data, inc=inc, r_min=r_min,
                                   r_max=r_max, vlsr=vlsr, chans=chans,
                                   kernel=kernel, min_SNR=min_SNR,
                                   detect_peaks_kwargs=detect_peaks_kwargs,
                                   force_opposite_sides=force_opposite_sides)
        if self.verbose:
            print("Done!")
        return surface(*_surf, chans=chans, rms=self.estimate_RMS(),
                       x0=x0, y0=y0, inc=inc, PA=PA, vlsr=vlsr, r_min=r_min,
                       r_max=r_max, data=data, masks=mask)

    def get_emission_surface_with_prior(self, prior_surface, nbeams=1.0,
                                        min_SNR=0.0):
        """
        If a surface prior is already given then we can use that to determine
        a mask to improve the peak detection.

        Args:
            prior_surface (surface instance): A previously derived ``suface``
                instance from which the velocity profile and emission height
                will be taken to define the new mask for the surface fitting.
            nbeams (optional[float]): The size of the convolution kernel in
                beam major FWHM that is used to broaden the mask. Larger values
                are more conservative and will take longer to converge.
            min_SNR (optional[float]): Specift a minimum SNR of the extracted
                points. Will used the RMS measured from the ``surface``.

        Returns:
            A ``disksurf.surface`` instance containing the extracted emission
            surface.
        """

        # Generate the surface based masks and SNR based masked. They will be
        # combined with ``np.logical_and``. TODO: Check that

        data = prior_surface.data
        chans = prior_surface.chans
        mask_near, mask_far = self.get_surface_mask(prior_surface,
                                                    nbeams,
                                                    min_SNR)

        # Find the peaks for just the near side, and then the back side and
        # concatenate the results into a new surface instance.
        # TODO: Would this be better to extract this and make a new function?

        print("Detecting peaks...")

        _surface = []
        inc_rad = np.radians(prior_surface.inc)
        for c_idx in range(data.shape[0]):

            # Check that the channel is in one of the channel ranges. If not,
            # we skip to the next channel index.

            c_idx_tot = c_idx + chans.min()
            if not any([ct[0] <= c_idx_tot <= ct[1] for ct in chans]):
                continue

            for x_idx in range(self.xaxis.size):

                # Check if there is any reigons to fit for this column. It
                # shouldn't matter which of the two masks we use as they should
                # only have unmasked regions if both have unmasked regions.

                points_near = np.any(mask_near[c_idx_tot, :, x_idx])
                points_far = np.any(mask_far[c_idx_tot, :, x_idx])
                if not points_near or not points_far:
                    continue

                # We cna skip a lot of the conditions from the standard
                # detect_peaks loop because we just use the mask to determine
                # if it's a channel to fit.
                # TODO: This could be vectorized relatively(?) easily.

                x_c = self.xaxis[x_idx]
                y_n_idx = np.nanargmax(np.where(mask_near[c_idx_tot, :, x_idx],
                                       data[c_idx, :, x_idx], np.nan))
                y_f_idx = np.nanargmax(np.where(mask_far[c_idx_tot, :, x_idx],
                                       data[c_idx, :, x_idx], np.nan))

                if data[c_idx, y_f_idx, x_idx] > data[c_idx, y_n_idx, x_idx]:
                    y_n, y_f = self.yaxis[y_n_idx], self.yaxis[y_f_idx]
                    Inu = data[c_idx, y_f_idx, x_idx]
                else:
                    y_n, y_f = self.yaxis[y_f_idx], self.yaxis[y_n_idx]
                    Inu = data[c_idx, y_n_idx, x_idx]

                # Use (x_c, y_n, y_f, vlsr) to calculate (y_c, r, z, v)
                # following Pinte et al. (2018).

                y_c = 0.5 * (y_f + y_n)
                r = np.hypot(x_c, (y_f - y_c) / np.cos(inc_rad))
                z = y_c / np.sin(inc_rad)
                v_chan = self.velax[c_idx_tot]
                v_int = (v_chan - prior_surface.vlsr) * r
                v_int /= x_c * np.sin(inc_rad)
                Tb = self.jybeam_to_Tb(Inu)

                # Add these values to the surface list. Set all the back side
                # values to NaNs.

                _surface += [[r, z, Inu, Tb, v_int, x_c, y_n, y_f, np.nan,
                              np.nan, np.nan, np.nan, np.nan, np.nan, v_chan]]
        print("Done")

        # Remove any non-finite values and return.

        _surface = np.squeeze(_surface).T
        return surface(*_surface[:, np.isfinite(_surface[2])],
                       chans=prior_surface.chans, rms=prior_surface.rms,
                       x0=prior_surface.x0, y0=prior_surface.y0,
                       inc=prior_surface.inc, PA=prior_surface.PA,
                       vlsr=prior_surface.vlsr, r_min=prior_surface.r_min,
                       r_max=prior_surface.r_max, data=data,
                       masks=[mask_near, mask_far])

    def get_emission_surface_iterative(self, prior_surface, N=5, nbeams=1.0,
                                       min_SNR=0.0):
        """
        Iteratively calculate the emission surface using ``N`` iterations. For
        both ``nbeams`` and ``min_SNR`` either a single value can be provided,
        and that value will be used for all iterations, or a list can be given
        to allow for a different value for each iteration. This is useful if
        you want to start with a large ``nbeams`` and gradually get smaller.

        Note: make sure the starting surface, ``prior_surface`` is reasonable
        so this does not diverge!

        Args:
            prior_surface (surface instance): A previously derived ``suface``
                instance from which the velocity profile and emission height
                will be taken to define the new mask for the surface fitting.
            nbeams (optional[float]): The size of the convolution kernel in
                beam major FWHM that is used to broaden the mask. Larger values
                are more conservative and will take longer to converge.
            min_SNR (optional[float]): Specift a minimum SNR of the extracted
                points. Will used the RMS measured from the ``surface``.

        Returns:
            A ``disksurf.surface`` instance containing the extracted emission
            surface.
        """
        _s = prior_surface
        nbeams = np.atleast_1d(nbeams)
        min_SNR = np.atleast_1d(min_SNR)
        for iter in range(N):
            print(f"Running iteration {iter+1}/{N}...")
            idx_a = iter % nbeams.size
            idx_b = iter % min_SNR.size
            _s = self.get_emission_surface_with_prior(prior_surface=_s,
                                                      nbeams=nbeams[idx_a],
                                                      min_SNR=min_SNR[idx_b])
        return _s

    def get_aligned_rotated_data(self, inc, PA, x0=0.0, y0=0.0, chans=None,
                                 r_min=None, r_max=None,
                                 get_keplerian_mask_kwargs=None):
        """
        Wrapper to get the aligned and rotated data ready for peak detection.

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
            get_keplerian_mask_kwargs (optional[dict]): A dictionary of values
                used to build a Keplerian mask. This requires at least the
                dynamical mass, ``mstar`` and the source distance, ``dist``.

        Returns:
            data
        """
        # Remove bad inclination:

        if inc == 0.0:
            raise ValueError("Cannot infer height with face on disk.")
        if self.verbose and abs(inc) < 10.0:
            print("WARNING: Inferences with close to face on disk are poor.")

        # Determine the spectral regin to fit.

        chans, data = self._get_velocity_clip_data(self.data.copy(), chans)

        # Align and rotate the data such that the major axis is parallel with
        # the x-axis. The red-shifted axis will be aligned with positive x
        # values. TODO: We can save this as a copy for user later for plotting
        # or repeated surface extractions.

        data = self._align_and_rotate_data(data=data, x0=x0, y0=y0, PA=PA)

        return chans, data

    # -- DATA MANIPULATION -- #

    def get_SNR_mask(self, surface=None, min_SNR=0.0):
        """
        Return a SNR based mask where pixels with intensities less than
        ``min_SNR * RMS`` are masked.

        Args:
            surface (optional[surface instance]): A previously derived
                ``suface`` instance.
            min_SNR (optional[float]): Specift a minimum SNR of the extracted
                points. Will used the RMS measured from the ``surface``.

        Return:
            SNR_mask.
        """
        if surface is None:
            data = self.data
            rms = self.estimate_RMS()
        else:
            data = surface.data
            rms = surface.rms
        return np.where(data >= min_SNR * rms, True, False)

    def get_surface_mask(self, surface, nbeams=1.0, min_SNR=0.0):
        """
        Calculate a mask based on a prior surface, ``surface``. Both the
        radial velocity profile and the emission surface will be used to
        calculate the expected isovelocity contours for the top side of the
        disk in each channel. These contours are then used to define a mask for
        the fitting of a new surface.

        The mask is initially a top hat function centered on the isovelocity
        contour, but can be broadened through the convolution of a 2D Gaussian
        kernel, the size of which is controlled with ``nbeams``.

        Note that ``data.shape != self.data.shape``.

        Args:
            surface (surface instance): A previously derived ``suface``
                instance from which the velocity profile and emission height
                will be taken to define the new mask for the surface fitting.
            nbeams (optional[float]): The size of the convolution kernel in
                beam major FWHM that is used to broaden the mask. Larger values
                are more conservative and will take longer to converge.
            min_SNR (optional[float]): Specift a minimum SNR of the extracted
                points. Will used the RMS measured from the ``surface``.

        Returns:
            mask_near, mask_far.
        """

        # Create an interpolatable emission surface to define the regions we
        # want to fit and an interpolatable rotation profile.
        # TODO: Check how we want to pass parameters to this function.

        z_func = surface.interpolate_parameter('z', method='binned')
        v_func = surface.interpolate_parameter('v', method='binned')

        # Based on the emission surface we produce a v0 map for the top side of
        # the disk. TODO: Verify the the choice of PA=90.0 is appropriate.

        r, phi, _ = self.disk_coords(inc=surface.inc, PA=90.0, z_func=z_func)
        v0 = v_func(r) * np.cos(phi) * np.sin(np.radians(surface.inc))
        v0 += surface.vlsr

        # Split the v0 map into front and back sides based on the change in v0
        # as a function of y. One side is always increasing, the other is
        # always decreasing.

        dv = np.sign(np.diff(v0, axis=0))
        dv = np.vstack([dv[0], dv])

        # Create a SNR mask so that can be included in the convolution.

        mask_snr = self.get_SNR_mask(surface, min_SNR)

        # Create a mask for the near side and the far side of the disk.

        print("Calculating masks...")

        mask_near, mask_far = [], []
        for cidx, velo in enumerate(self.velax):

            # Skip the unused channels.

            if not any([ct[0] <= cidx <= ct[1] for ct in surface.chans]):
                mask_near += [np.zeros(self.data[0].shape).astype(bool)]
                mask_far += [np.zeros(self.data[0].shape).astype(bool)]
                continue

            # Find the absolute deviation in order to define the radial range
            # of the mask. A broader tolerance will lead to the masks extending
            # to larger radii. TODO: Check how this is impacted with different
            # inclinations of disks.

            absolute_deviation = np.nanmin(abs(v0 - velo), axis=0)
            radial_mask = absolute_deviation <= 0.25 * self.chan

            # Masks are a top hat function with a width of the beam across the
            # isovelocity contour, then convolved with a Gaussian kernel with a
            # FWHM equal to that of the beam major axis.
            # TODO: Check what values or defaults we want here.

            isovelocity_t = abs(np.where(dv > 0, v0, -1e10) - velo)
            isovelocity_t = np.nanargmin(isovelocity_t, axis=0)
            isovelocity_b = abs(np.where(dv < 0, v0, -1e10) - velo)
            isovelocity_b = np.nanargmin(isovelocity_b, axis=0)

            mask_t = np.where(radial_mask, self.yaxis[isovelocity_t], np.nan)
            mask_t = abs(self.yaxis[:, None] - mask_t[None, :]) <= self.bmaj
            mask_b = np.where(radial_mask, self.yaxis[isovelocity_b], np.nan)
            mask_b = abs(self.yaxis[:, None] - mask_b[None, :]) <= self.bmaj

            mask_t = np.logical_and(mask_snr[cidx], mask_t)
            mask_b = np.logical_and(mask_snr[cidx], mask_b)

            kernel = Gaussian2DKernel(nbeams * self.bmaj / self.dpix / 2.355)
            mask_t = convolve(mask_t, kernel) >= 0.1
            mask_b = convolve(mask_b, kernel) >= 0.1

            # We want to remove regions where the masks overlap (generally at
            # the disk edge along the major axis) and where there is only a top
            # or a bottom mask.

            overlap = np.logical_and(mask_t, mask_b)
            mask_t = np.where(~overlap, mask_t, False)
            mask_b = np.where(~overlap, mask_b, False)

            both_masks = np.logical_and(np.any(mask_t, axis=0),
                                        np.any(mask_b, axis=0))
            mask_t = np.where(both_masks[None, :], mask_t, False)
            mask_b = np.where(both_masks[None, :], mask_b, False)

            # Add the masks to the arrays.

            mask_near += [mask_t]
            mask_far += [mask_b]

        # Check the shapes of the arrays. Note that the shame of the masks
        # are the same as the full data (self.data), while the rotated and
        # aligned data (data) has been clipped in velocity space.

        mask_near, mask_far = np.squeeze(mask_near), np.squeeze(mask_far)
        assert mask_near.shape == mask_far.shape == self.data.shape
        return mask_near, mask_far

    def get_keplerian_mask(self, x0, y0, inc, PA, mstar, vlsr, dist, r_min=0.0,
                           r_max=None, width=2.0, smooth=None, tolerance=1e-4):
        """
        Produce a Keplerian mask for the data.

        Args:
            x0 (float): Disk offset along the x-axis in [arcsec].
            y0 (float): Disk offset along the y-axis in [arcsec].
            inc (float): Disk inclination in [degrees].
            PA (float): Disk position angle in [degrees].
            mstar (float): Stellar mass in [Msun].
            vlsr (float): Systemic velocity in [m/s].
            dist (float): Source distance in [pc].
            r_min (optional[float]): Inner radius in [arcsec].
            r_max (optional[float]): Outer radius in [arcsec].
            width (optional[float]): The spectral 'width' of the mask as a
                fraction of the channel spacing.
            smooth (optional[float]): Apply a convolution with a 2D Gaussian
                with a FWHM of ``smooth`` to broaden the mask. By default this
                is four times the beam FWHM. If no smoothing is desired, set
                this to ``0.0``.
            tolerance (optional[float]): The minimum value (between 0 and 1) to
                consider part of the mask after convolution.

        Returns:
            A 3D array describing the mask with either ``True`` or ``False``.
        """

        # Generate line-of-sight velocity profile.

        vkep = self.keplerian(x0=x0, y0=y0, inc=inc, PA=PA,
                              mstar=mstar, vlsr=vlsr,
                              dist=dist)

        # Apply a radial mask. This will be in addition to the simple r_min and
        # r_max cuts applied in `detect_peaks`.

        rvals = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA)[0]
        r_max = rvals.max() if r_max is None else r_max
        assert r_min < r_max, "r_min >= r_max"
        rmask = np.logical_and(rvals >= r_min, rvals <= r_max)
        vkep = np.where(rmask, vkep, np.nan)

        # Generate the mask in 3D.

        mask = abs(self.velax[:, None, None] - vkep[None, :, :])
        mask = np.where(mask <= width * self.chan, 1.0, 0.0)

        # Smooth the mask with a 2D Gaussian.

        smooth = 4.0 * self.bmaj if smooth is None else smooth
        if smooth > 0.0:
            print("Smoothing mask. May take a while...")
            from scipy.ndimage import gaussian_filter
            kernel = smooth / self.dpix / 2.355
            mask = np.array([gaussian_filter(c, kernel) for c in mask])
            mask = np.where(mask >= tolerance, 1.0, 0.0)
        assert mask.shape == self.data.shape, "mask.shape != data.shape"
        return mask.astype('bool')

    def _get_velocity_clip_data(self, data, chans=None):
        """Clip the data based on a provided channel range."""
        if chans is None:
            chans = [0, data.shape[0] - 1]
        chans = np.atleast_2d(chans).astype('int')
        if chans.min() < 0:
            raise ValueError("`chans` has negative values.")
        if chans.max() >= data.shape[0]:
            raise ValueError("`chans` extends beyond the number of channels.")
        return chans, data.copy()[chans.min():chans.max()+1]

    def _align_and_rotate_data(self, data, x0=None, y0=None, PA=None):
        """Align and rotate the data."""
        if x0 != 0.0 or y0 != 0.0:
            if self.verbose:
                print("Centering data cube...")
            x0_pix = x0 / self.dpix
            y0_pix = y0 / self.dpix
            data = observation._shift_center(data, x0_pix, y0_pix)
        if PA != 90.0 and PA != 270.0:
            if self.verbose:
                print("Rotating data cube...")
            data = observation._rotate_image(data, PA)
        return data

    def _detect_peaks(self, data, inc, r_min, r_max, vlsr, chans, min_SNR=5.0,
                      kernel=None, return_back=True, detect_peaks_kwargs=None,
                      force_opposite_sides=True):
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
                    # disk, but that's more conservative anyway. There is the
                    # `force_opposite_sides` switch to skip this if necessary.

                    if y_f * y_n > 0.0 and force_opposite_sides:
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
                        if y_fb * y_nb > 0.0 and force_opposite_sides:
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

                # Transform the channel velocity to the true velocity following
                # Eqn. 3 from Pinte et al. (2018).

                v_int = (v - vlsr) * r / x_c / np.sin(inc_rad)

                # Bring together all the points needed for a surface instance.

                peaks = [r, z, Inu, self.jybeam_to_Tb(Inu),
                         v_int, x_c, y_n, y_f, rb, zb, Inub,
                         self.jybeam_to_Tb(Inu), y_nb, y_fb, v]
                _surface += [peaks]

        # Remove any non-finite values and return.

        _surface = np.squeeze(_surface).T
        return _surface[:, np.isfinite(_surface[2])]

    def get_integrated_spectrum(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0,
                                r_max=None):
        """
        Calculate the integrated spectrum over a specified spatial region. The
        uncertainty is calculated assuming the spatially correlation is given
        by elliptical beams.

        Args:
            x0 (optional[float]): Right Ascension offset in [arcsec].
            y0 (optional[float]): Declination offset in [arcsec].
            inc (optional[float]): Disk inclination in [deg].
            PA (optional[float]): Disk position angle in [deg].
            r_max (optional[float]): Radius to integrate out to in [arcsec].

        Returns:
            The integrated intensity, ``spectrum``, and associated uncertainty,
            ``uncertainty``, in [Jy].
        """
        rr = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA)[0]
        r_max = rr.max() if r_max is None else r_max
        nbeams = np.where(rr <= r_max, 1, 0).sum() / self.pix_per_beam
        spectrum = np.array([np.nansum(c[rr <= r_max]) for c in self.data])
        spectrum *= self.beams_per_pix
        uncertainty = np.sqrt(nbeams) * self.estimate_RMS()
        return spectrum, uncertainty

    @staticmethod
    def _rotate_image(data, PA):
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
    def _shift_center(data, x0, y0):
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
    def _powerlaw(r, z0, q, r_cavity=0.0):
        """Standard power law profile."""
        return z0 * np.clip(r - r_cavity, a_min=0.0, a_max=None)**q

    @staticmethod
    def _tapered_powerlaw(r, z0, q, r_taper=np.inf, q_taper=1.0, r_cavity=0.0):
        """Exponentially tapered power law profile."""
        rr = np.clip(r - r_cavity, a_min=0.0, a_max=None)
        f = observation._powerlaw(rr, z0, q)
        return f * np.exp(-(rr / r_taper)**q_taper)

    # -- PLOTTING FUNCTIONS -- #

    def plot_channels(self, chans=None, velocities=None, return_fig=False,
                      get_keplerian_mask_kwargs=None):
        """
        Plot the channels within the channel range or velocity range. Only one
        of ``chans`` or ``velocities`` can be specified. If neither is
        specified, all channels are plotted which may take some time for large
        data cubes.

        Args:
            chans (optional[tuple]): A tuple containing the index of the first
                and last channel to plot. Cannot be specified if ``velocities``
                is also specified.
            velocities (optional[tuple]): A tuple containing the velocity of
                the first and last channel to plot in [m/s]. Cannot be
                specified if ``chans`` is also specified.
            return_fig (optional[bool]): Whether to return the Matplotlib
                figure.
            get_keplerian_mask_kwargs (optional[dict]): A dictionary of arguments
                to pass to ``get_keplerian_mask`` such that the mask outline can
                be overlaid.

        Returns:
            If ``return_fig=True``, the Matplotlib figure used for plotting.
        """
        from matplotlib.ticker import MaxNLocator
        import matplotlib.pyplot as plt

        # Calculate the Keplerian mask.

        if get_keplerian_mask_kwargs is not None:
            mask = self.get_keplerian_mask(**get_keplerian_mask_kwargs)
        else:
            mask = None

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
            if mask is not None:
                ax.contour(self.xaxis, self.yaxis, mask[chans[0]+a], [0.5],
                           colors='orangered', linestyles='--', linewidths=0.5)
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

    def plot_integrated_spectrum(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0,
                                 r_max=None, return_fig=False):
        """
        Plot the integrated spectrum integrated over a spatial region.

        Args:
            x0 (optional[float]): Right Ascension offset in [arcsec].
            y0 (optional[float]): Declination offset in [arcsec].
            inc (optional[float]): Disk inclination in [deg].
            PA (optional[float]): Disk position angle in [deg].
            r_max (optional[float]): Radius to integrate out to in [arcsec].

        Returns:
            If ``return_fig=True``, the Matplotlib figure used for plotting.
        """
        x = self.velax.copy() / 1e3
        y, dy = self.get_integrated_spectrum(x0, y0, inc, PA, r_max)
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
        if return_fig:
            return fig

    def plot_isovelocities(self, surface, mstar, vlsr, dist, side='both',
                           reflect=True, smooth=None, return_fig=False):
        """
        Plot the isovelocity contours for the given emission surface. This will
        only overlay contours on the channels used for the extraction of the
        emission surface.

        TODO: Rather than an analytical profile, use the rolling statistic or
        binned profile.

        Args:
            surface (surface instance): The extracted emission surface.
            mstar (float): The stellar mass in [Msun].
            vlsr (float): The systemic velocity in [m/s].
            dist (float): The source distance in [pc].
            side (optional[str]): The emission side to plot, must be either
                ``'both'``, ``'front'`` or ``'back'``.
            reflect (optional[bool]): Whether to reflect the back side of the
                disk about the midplane. Default is ``False``.
            smooth (optional[int]): If provided, smooth the emission surface
                with a Hanning kernel with a width of ``smooth``. Typically
                values of 3 or 5 are sufficient for plotting purposes.
            return_fig (optional[bool]): If no axis is provided, whether to
                return the Matplotlib figure. The axis can then be accessed
                through ``fig.axes[0]``.

        Returns:
            If ``return_fig=True``, the Matplotlib figure used for plotting.
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
        Plot the peak locations used to calculate the emission surface on
        channel maps. This will use the channels used for the extraction of the
        emission surface.

        Args:
            surface (surface instance): The extracted surface returned from
                ``get_emission_surface``.
            side (Optional[str]): Side to plot. Must be ``'front'``, ``'back'``
                or ``'both'``. Defaults to ``'both'``.
            return_fig (Optional[bool]): Whether to return the Matplotlib
                figure. Defaults to ``True``.

        Returns:
            If ``return_fig=True``, the Matplotlib figure used for plotting.
        """
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        velocities = self.velax[surface.chans.min():surface.chans.max()+1]
        nrows = np.ceil(velocities.size / 5).astype(int)
        fig, axs = plt.subplots(ncols=5, nrows=nrows, figsize=(11, 2*nrows+1),
                                constrained_layout=True)

        velax = self.velax[surface.chans.min():surface.chans.max()+1]
        data = surface.data.copy()

        for vv, ax in zip(velocities, axs.flatten()):

            channel = data[abs(velax - vv).argmin()]

            ax.imshow(channel, origin='lower', extent=self.extent,
                      cmap='binary_r', vmin=0.0, vmax=0.75*data.max())

            # Plot the back side.

            if side.lower() in ['back', 'both']:
                toplot = surface.v_chan(side='back') == vv
                ax.scatter(surface.x(side='back')[toplot],
                           surface.y(side='back', edge='far')[toplot],
                           lw=0.0, color='r', marker='.')
                ax.scatter(surface.x(side='back')[toplot],
                           surface.y(side='back', edge='near')[toplot],
                           lw=0.0, color='r', marker='.')

            # Plot the front side.

            if side.lower() in ['front', 'both']:
                toplot = surface.v_chan(side='front') == vv
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

        # Remove unused axes.

        if axs.size != velocities.size:
            for ax in axs.flatten()[-(axs.size - velocities.size):]:
                ax.axis('off')

        # Returns.

        if return_fig:
            return fig

    def plot_mask(self, surface, nbeams=1.0, return_fig=False):
        """
        Args:
            surface (surface instance): The extracted surface returned from
                ``get_emission_surface``.
            nbeams:
            return_fig (optional[bool]): Whether to return the Matplotlib
                figure. Defaults to ``True``.

        Returns:
            If ``return_fig=True``, the Matplotlib figure used for plotting.
        """
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        # Define the channel map grid.

        velocities = self.velax[surface.chans.min():surface.chans.max()+1]
        nrows = np.ceil(velocities.size / 5).astype(int)
        fig, axs = plt.subplots(ncols=5, nrows=nrows, figsize=(11, 2*nrows+1),
                                constrained_layout=True)

        # Grab the data, velocity axis and the masks.

        velax = self.velax[surface.chans.min():surface.chans.max()+1]

        for vv, ax in zip(velocities, axs.flatten()):

            c_idx = abs(velax - vv).argmin()
            m_idx = abs(self.velax - vv).argmin()

            ax.imshow(surface.data[c_idx], origin='lower', extent=self.extent,
                      cmap='binary_r', vmin=0.0, vmax=0.75*surface.data.max())

            ax.contour(self.xaxis, self.yaxis, surface.mask_near[m_idx],
                       colors='r')
            ax.contour(self.xaxis, self.yaxis, surface. mask_far[m_idx],
                       colors='b')

            # Gentrification.

            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.grid(ls='--', lw=1.0, alpha=0.2)

            if ax != axs[-1, 0]:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            else:
                ax.set_xlabel('Offset (arcsec)')
                ax.set_ylabel('Offset (arcsec)')

        # Remove unused axes.

        if axs.size != velocities.size:
            for ax in axs.flatten()[-(axs.size - velocities.size):]:
                ax.axis('off')

        # Returns.

        if return_fig:
            return fig

    def plot_temperature(self, surface, side='both', reflect=False,
                         masked=True, ax=None, return_fig=False):
        r"""
        Plot the temperature structure using the provided surface instance.
        Note that the brightness temperature only provides a good measure of
        the true gas temperature when the lines are optically thick such that
        :math:`\tau \gtrsim 5`.

        Args:
            surface (surface instance): The extracted emission surface.
            side (optional[str]): The emission side to plot, must be either
                ``'both'``, ``'front'`` or ``'back'``.
            reflect (optional[bool]): Whether to reflect the back side of the
                disk about the midplane. Default is ``False``.
            masked (optional[bool]): Whether to plot the masked points, the
                default, or all extracted points.
            ax (optional[axes instance]): The Matplolib axis to use for
                plotting. If none is provided, one will be generated. If an
                axis is provided, the same color scaling will be used.
            return_fig (optional[bool]): If no axis is provided, whether to
                return the Matplotlib figure. The axis can then be accessed
                through ``fig.axes[0]``.

        Returns:
            If ``return_fig=True``, the Matplotlib figure used for plotting.
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
            Tb = np.concatenate([Tb, surface.T(side='front', masked=masked)])
        if side.lower() in ['back', 'both']:
            r = np.concatenate([r, surface.r(side='back', masked=masked)])
            _z = surface.z(side='back', reflect=reflect, masked=masked)
            z = np.concatenate([z, _z])
            Tb = np.concatenate([Tb, surface.T(side='back', masked=masked)])
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
