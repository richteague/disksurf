import matplotlib.pyplot as plt
import numpy as np


class surface(object):
    """
    A container for the emission surface returned by ``detect_peaks``. This
    class has been designed to be created by the ``get_emission_surface``
    function and not by the user.

    Args:
        r_f (array): Radial position of the front surface in [arcsec].
        z_f (array): Vertical position of the front surface in [arcsec].
        I_f (array): Intensity along the front surface in [Jy/beam].
        T_f (array): Brightness temperature along the front surface in [K].
        v (array): Intrinsic velocity in [m/s].
        x (array): Distance along the major axis the point was extracted in
            [arcsec].
        y_n (array): Distance along the minor axis of the near peak for the
            front surface in [arcsec].
        y_f (array): Distance along the minor axis of the far peak for the
            front surface in [arcsec].
        r_b (array): Radial position of the back surface in [arcsec].
        z_b (array): Vertical position of the back surface in [arcsec].
        I_b (array): Intensity along the back surface in [Jy/beam].
        T_b (array): Brightness temperature along the back surface in [K].
        y_n_b (array): Distance along the minor axis of the near peak for the
            back surface in [arcsec].
        y_f_b (array): Distance along the minor axis of the far peak for the
            back surface in [arcsec].
        v_chan (array): The velocity of the channel the point was extracted
            from in [m/s].
        chans (tuple): A tuple of the first and last channels used for the
            emission surface extraction.
        rms (float): Noise in the cube in [Jy/beam].
        x0 (float): Right ascencion offset used in the emission surface
            extraction in [arcsec].
        y0 (float): Declination offset used in the emission surface extraction
            in [arcsec].
        inc (float): Inclination of the disk used in the emission surface
            extraction in [deg].
        PA (float): Position angle of the disk used in the emission surface
            extraction in [deg].
        vlsr (float): Systemic velocity of the system in [m/s].
        r_min (float): Minimum disk-centric radius used in the emission surface
            extraction in [arcsec].
        r_max (array): Maximum disk-centric radius used in the emission surface
            extraction in [arcsec].
        data (array): The data used to extract the emission surface in
            [Jy/beam].
        masks (array): A tuple of the near and far masks used to extract the
            emission surface [bool].
    """

    def __init__(self, r_f, z_f, I_f, T_f, v, x, y_n, y_f, r_b, z_b, I_b,
                 T_b, y_n_b, y_f_b, v_chan, chans, rms, x0, y0, inc, PA, vlsr,
                 r_min, r_max, data, masks):

        # Parameters used to extract the emission surface.

        self.inc = inc
        self.PA = PA
        self.x0 = x0
        self.y0 = y0
        self.vlsr = vlsr
        self.chans = chans
        self.r_min = r_min
        self.r_max = r_max
        self.rms = rms
        self.data = data

        # Split the mask into near and far masks. If there is only one mask we
        # assume the same mask for near and far.

        masks = np.squeeze(masks)
        if masks.ndim == 4:
            self.mask_near = masks[0]
            self.mask_far = masks[1]
        elif masks.ndim == 3:
            self.mask_near = masks
            self.mask_far = masks
        else:
            self.mask_near = np.ones(self.data.shape).astype('bool')
            self.mask_far = np.ones(self.data.shape).astype('bool')

        # Properties of the emission surface.

        idx = np.argsort(r_f)
        self._r_f = np.squeeze(r_f)[idx]
        self._z_f = np.squeeze(z_f)[idx]
        self._I_f = np.squeeze(I_f)[idx]
        self._T_f = np.squeeze(T_f)[idx]
        self._r_b = np.squeeze(r_b)[idx]
        self._z_b = np.squeeze(z_b)[idx]
        self._I_b = np.squeeze(I_b)[idx]
        self._T_b = np.squeeze(T_b)[idx]

        self._v = np.squeeze(v)[idx]
        self._x = np.squeeze(x)[idx]

        self._y_n_f = np.squeeze(y_n)[idx]
        self._y_f_f = np.squeeze(y_f)[idx]
        self._y_n_b = np.squeeze(y_n_b)[idx]
        self._y_f_b = np.squeeze(y_f_b)[idx]
        self._v_chan = np.squeeze(v_chan)[idx]
        self.reset_mask()

    def r(self, side='front', masked=True):
        """
        Radial cylindrical coordinate in [arcsec].

        Args:
            side (optional[str]): Side of the disk. Must be ``'front'``,
                ``'back'`` or ``'both'``. Defaults to ``'both'``.
            masked (optional[bool]): Whether to return only the masked points,
                the default, or all points.

        Returns:
            Radial cylindrical coordinates in [arcsec].
        """
        if side not in ['front', 'back', 'both']:
            raise ValueError(f"Unknown `side` value {side}.")
        r = np.empty(1)
        if side in ['front', 'both']:
            if masked:
                r_tmp = self._r_f[self._mask_f].copy()
            else:
                r_tmp = self._r_f.copy()
            r = np.concatenate([r, r_tmp])
        if side in ['back', 'both']:
            if masked:
                r_tmp = self._r_b[self._mask_b].copy()
            else:
                r_tmp = self._r_b.copy()
            r = np.concatenate([r, r_tmp])
        return np.squeeze(r[1:])

    def z(self, side='front', reflect=False, masked=True):
        """
        Vertical cylindrical coordinate in [arcsec].

        Args:
            side (optional[str]): Side of the disk. Must be ``'front'``,
                ``'back'`` or ``'both'``. Defaults to ``'both'``.
            reflect (optional[bool]): Whether to reflect the backside points
                about the midplane. Defaults to ``False``.
            masked (optional[bool]): Whether to return only the masked points,
                the default, or all points.

        Returns:
            Vertical cylindrical coordinate in [arcsec].
        """
        if side not in ['front', 'back', 'both']:
            raise ValueError(f"Unknown `side` value {side}.")
        z = np.empty(1)
        if side in ['front', 'both']:
            if masked:
                z_tmp = self._z_f[self._mask_f].copy()
            else:
                z_tmp = self._z_f.copy()
            z = np.concatenate([z, z_tmp])
        if side in ['back', 'both']:
            if masked:
                z_tmp = self._z_b[self._mask_b].copy()
            else:
                z_tmp = self._z_b.copy()
            z = np.concatenate([z, -z_tmp if reflect else z_tmp])
        return np.squeeze(z[1:])

    def I(self, side='front', masked=True):
        """
        Intensity at the (r, z) coordinate in [Jy/beam].

        Args:
            side (optional[str]): Side of the disk. Must be ``'front'``,
                ``'back'`` or ``'both'``. Defaults to ``'both'``.
            masked (optional[bool]): Whether to return only the masked points,
                the default, or all points.

        Returns:
            Intensity at the (r, z) coordinate in [Jy/beam].
        """
        if side not in ['front', 'back', 'both']:
            raise ValueError(f"Unknown `side` value {side}.")
        i = np.empty(1)
        if side in ['front', 'both']:
            if masked:
                i_tmp = self._I_f[self._mask_f].copy()
            else:
                i_tmp = self._I_f.copy()
            i = np.concatenate([i, i_tmp])
        if side in ['back', 'both']:
            if masked:
                i_tmp = self._I_b[self._mask_b].copy()
            else:
                i_tmp = self._I_b.copy()
            i = np.concatenate([i, i_tmp])
        return np.squeeze(i[1:])

    def T(self, side='front', masked=True):
        """
        Brightness temperature at the (r, z) coordinate in [K].

        Args:
            side (optional[str]): Side of the disk. Must be ``'front'``,
                ``'back'`` or ``'both'``. Defaults to ``'both'``.
            masked (optional[bool]): Whether to return only the masked points,
                the default, or all points.

        Returns:
            Brightness temperature at the (r, z) coordinate in [K].
        """
        if side not in ['front', 'back', 'both']:
            raise ValueError(f"Unknown `side` value {side}.")
        T = np.empty(1)
        if side in ['front', 'both']:
            if masked:
                T_tmp = self._T_f[self._mask_f].copy()
            else:
                T_tmp = self._T_f.copy()
            T = np.concatenate([T, T_tmp])
        if side in ['back', 'both']:
            if masked:
                T_tmp = self._T_b[self._mask_b].copy()
            else:
                T_tmp = self._T_b.copy()
            T = np.concatenate([T, T_tmp])
        return np.squeeze(T[1:])

    def v(self, side='front', masked=True):
        """
        Intrinsic velocity at the (r, z) coordinate in [m/s].

        Args:
            side (optional[str]): Side of the disk. Must be ``'front'``,
                ``'back'`` or ``'both'``. Defaults to ``'both'``.
            masked (optional[bool]): Whether to return only the masked points,
                the default, or all points.

        Returns:
            Intrinsic velocity at the (r, z) coordinate in [m/s].
        """
        if side not in ['front', 'back', 'both']:
            raise ValueError(f"Unknown `side` value {side}.")
        v = np.empty(1)
        if side in ['front', 'both']:
            if masked:
                v_tmp = self._v[self._mask_f].copy()
            else:
                v_tmp = self._v.copy()
            v = np.concatenate([v, v_tmp])
        if side in ['back', 'both']:
            if masked:
                v_tmp = self._v[self._mask_b].copy()
            else:
                v_tmp = self._v.copy()
            v = np.concatenate([v, v_tmp])
        return np.squeeze(v[1:])

    def x(self, side='front', masked=True):
        """
        RA offset that the (r, z) coordinate was extracted in [arcsec].

        Args:
            side (optional[str]): Side of the disk. Must be ``'front'``,
                ``'back'`` or ``'both'``. Defaults to ``'both'``.
            masked (optional[bool]): Whether to return only the masked points,
                the default, or all points.

        Returns:
            RA offset that the (r, z) coordinate was extracted in [arcsec].
        """
        if side not in ['front', 'back', 'both']:
            raise ValueError(f"Unknown `side` value {side}.")
        x = np.empty(1)
        if side in ['front', 'both']:
            if masked:
                x_tmp = self._x[self._mask_f].copy()
            else:
                x_tmp = self._x.copy()
            x = np.concatenate([x, x_tmp])
        if side in ['back', 'both']:
            if masked:
                x_tmp = self._x[self._mask_b].copy()
            else:
                x_tmp = self._x.copy()
            x = np.concatenate([x, x_tmp])
        return np.squeeze(x[1:])

    def y(self, side='front', edge='near', masked=True):
        """
        Dec offset that the (r, z) coordinate was extracted in [arcsec].

        Args:
            side (optional[str]): Side of the disk. Must be ``'front'``,
                ``'back'`` or ``'both'``. Defaults to ``'both'``.
            edge (optional[str]): Which of the edges to return, either the
                ``'near'`` or ``'far'`` edge.
            masked (optional[bool]): Whether to return only the masked points,
                the default, or all points.

        Returns:
            Dec offset that the (r, z) coordinate was extracted in [arcsec].
        """
        if side not in ['front', 'back', 'both']:
            raise ValueError(f"Unknown `side` value {side}.")
        if edge not in ['near', 'far']:
            raise ValueError(f"Unknown `edge` value {edge}.")
        y = np.empty(1)
        if side in ['front', 'both']:
            if edge == 'near':
                if masked:
                    y_tmp = self._y_n_f[self._mask_f].copy()
                else:
                    y_tmp = self._y_n_f.copy()
                y = np.concatenate([y, y_tmp])
            else:
                if masked:
                    y_tmp = self._y_f_f[self._mask_f].copy()
                else:
                    y_tmp = self._y_f_f.copy()
                y = np.concatenate([y, y_tmp])
        if side in ['back', 'both']:
            if edge == 'near':
                if masked:
                    y_tmp = self._y_n_b[self._mask_b].copy()
                else:
                    y_tmp = self._y_n_b.copy()
                y = np.concatenate([y, y_tmp])
            else:
                if masked:
                    y_tmp = self._y_f_b[self._mask_b].copy()
                else:
                    y_tmp = self._y_f_b.copy()
                y = np.concatenate([y, y_tmp])
        return np.squeeze(y[1:])

    def v_chan(self, side='front', masked=True):
        """
        Channel velocity that the (r, z) coordinate was extracted at in [m/s].

        Args:
            side (optional[str]): Side of the disk. Must be ``'front'``,
                ``'back'`` or ``'both'``. Defaults to ``'both'``.
            masked (optional[bool]): Whether to return only the masked points,
                the default, or all points.

        Returns:
            Velocity that the (r, z) coordinate was extracted at in [m/s].
        """
        if side not in ['front', 'back', 'both']:
            raise ValueError(f"Unknown `side` value {side}.")
        v = np.empty(1)
        if side in ['front', 'both']:
            if masked:
                v_tmp = self._v_chan[self._mask_f].copy()
            else:
                v_tmp = self._v_chan.copy()
            v = np.concatenate([v, v_tmp])
        if side in ['back', 'both']:
            if masked:
                v_tmp = self._v_chan[self._mask_b].copy()
            else:
                v_tmp = self._v_chan.copy()
            v = np.concatenate([v, v_tmp])
        return np.squeeze(v[1:])

    def zr(self, side='front', reflect=True, masked=True):
        """
        Inverse aspect ratio (height divided by radius) of the emission
        surface.

        Args:
            side (optional[str]): Side of the disk. Must be ``'front'``,
                ``'back'`` or ``'both'``. Defaults to ``'both'``.
            reflect (optional[bool]): Whether to reflect the backside points
                about the midplane. Defaults to ``False``.
            masked (optional[bool]): Whether to return only the masked points,
                the default, or all points.

        Returns:
            Inverse aspect ratio of the emission surface.
        """
        return self.z(side, reflect, masked) / self.r(side, masked)

    def SNR(self, side='front', masked=True):
        """
        Signal-to-noise ratio for each coordinate.

        Args:
            side (optional[str]): Side of the disk. Must be ``'front'``,
                ``'back'`` or ``'both'``. Defaults to ``'both'``.
            masked (optional[bool]): Whether to return only the masked points,
                the default, or all points.

        Returns:
            Signal-to-noise ratio for each coordinate.
        """
        return self.I(side, masked) / self.rms

    def reset_mask(self, side='both'):
        """
        Reset the mask.

        Args:
            side (optional[str]): Side of the disk. Must be ``'front'``,
                ``'back'`` or ``'both'``. Defaults to ``'both'``.
        """
        if side.lower() == 'front':
            self._mask_f = np.isfinite(self._z_f).astype('bool')
            self._mask_f *= np.isfinite(self._I_f).astype('bool')
        elif side.lower() == 'back':
            self._mask_b = np.isfinite(self._z_b).astype('bool')
            self._mask_b *= np.isfinite(self._I_b).astype('bool')
        elif side.lower() == 'both':
            self._mask_f = np.isfinite(self._z_f).astype('bool')
            self._mask_f *= np.isfinite(self._I_f).astype('bool')
            self._mask_b = np.isfinite(self._z_b).astype('bool')
            self._mask_b *= np.isfinite(self._I_b).astype('bool')
        else:
            raise ValueError(f"Unknown `side` value {side}.")

    @property
    def data_aligned_rotated_key(self):
        return (self.x0, self.y0, self.PA, self.chans.min(), self.chans.max())

    def mask_surface(self, side='front', reflect=False, min_r=None, max_r=None,
                     min_z=None, max_z=None, min_zr=None, max_zr=None,
                     min_I=None, max_I=None, min_v=None, max_v=None,
                     min_SNR=None, max_SNR=None, RMS=None):
        """
        Mask the surface based on simple cuts to the parameters.

        Args:
            min_r (optional[float]): Minimum radius in [arcsec].
            max_r (optional[float]): Maximum radius in [arcsec].
            min_z (optional[float]): Minimum emission height in [arcsec].
            max_z (optional[float]): Maximum emission height in [arcsec].
            min_zr (optional[float]): Minimum z/r ratio.
            max_zr (optional[float]): Maximum z/r ratio.
            min_Inu (optional[float]): Minumum intensity in [Jy/beam].
            max_Inu (optional[float]): Maximum intensity in [Jy/beam].
            min_v (optional[float]): Minimum velocity in [m/s].
            max_v (optional[float]): Maximum velocity in [m/s].
            min_snr (optional[float]): Minimum SNR ratio.
            max_snr (optional[float]): Maximum SNR ratio.
            RMS (optional[float]): Use this RMS value in place of the
                ``self.rms`` value for calculating the SNR masks.
        """

        # Minimum or maximum radius value.

        if min_r is not None or max_r is not None:
            if side in ['front', 'both']:
                r = self.r(side='front', masked=False)
                _min_r = np.nanmin(r) if min_r is None else min_r
                _max_r = np.nanmax(r) if max_r is None else max_r
                mask = np.logical_and(r >= _min_r, r <= _max_r)
                self._mask_f *= mask
            if side in ['back', 'both']:
                r = self.r(side='back', masked=False)
                _min_r = np.nanmin(r) if min_r is None else min_r
                _max_r = np.nanmax(r) if max_r is None else max_r
                mask = np.logical_and(r >= _min_r, r <= _max_r)
                self._mask_b *= mask

        # Minumum or maxium emission height.

        if min_z is not None or max_z is not None:
            if side in ['front', 'both']:
                z = self.z(side='front', masked=False)
                _min_z = np.nanmin(z) if min_z is None else min_z
                _max_z = np.nanmax(z) if max_z is None else max_z
                mask = np.logical_and(z >= _min_z, z <= _max_z)
                self._mask_f *= mask
            if side in ['back', 'both']:
                z = self.z(side='back', reflect=reflect, masked=False)
                _min_z = np.nanmin(z) if min_z is None else min_z
                _max_z = np.nanmax(z) if max_z is None else max_z
                mask = np.logical_and(z >= _min_z, z <= _max_z)
                self._mask_b *= mask

        # Minimum or maximum emission height aspect ratio.

        if min_zr is not None or max_zr is not None:
            if side in ['front', 'both']:
                zr = self.zr(side='front', masked=False)
                _min_zr = np.nanmin(zr) if min_zr is None else min_zr
                _max_zr = np.nanmax(zr) if max_zr is None else max_zr
                mask = np.logical_and(zr >= _min_zr, zr <= _max_zr)
                self._mask_f *= mask
            if side in ['back', 'both']:
                zr = self.zr(side='back', reflect=reflect, masked=False)
                _min_zr = np.nanmin(zr) if min_zr is None else min_zr
                _max_zr = np.nanmax(zr) if max_zr is None else max_zr
                mask = np.logical_and(zr >= _min_zr, zr <= _max_zr)
                self._mask_b *= mask

        # Minimum or maximum intensity.

        if min_I is not None or max_I is not None:
            if side in ['front', 'both']:
                _I = self.I(side='front', masked=False)
                _min_I = np.nanmin(_I) if min_I is None else min_I
                _max_I = np.nanmax(_I) if max_I is None else max_I
                mask = np.logical_and(_I >= _min_I, _I <= _max_I)
                self._mask_f *= mask
            if side in ['back', 'both']:
                _I = self.I(side='back', masked=False)
                _min_I = np.nanmin(_I) if min_I is None else min_I
                _max_I = np.nanmax(_I) if max_I is None else max_I
                mask = np.logical_and(_I >= _min_I, _I <= _max_I)
                self._mask_b *= mask

        # Minimum or maximum velocity.

        if min_v is not None or max_v is not None:
            if side in ['front', 'both']:
                v = self.v(side='front', masked=False)
                _min_v = np.nanmin(v) if min_v is None else min_v
                _max_v = np.nanmax(v) if max_v is None else max_v
                mask = np.logical_and(v >= _min_v, v <= _max_v)
                self._mask_f *= mask
            if side in ['back', 'both']:
                v = self.v(side='back', masked=False)
                _min_v = np.nanmin(v) if min_v is None else min_v
                _max_v = np.nanmax(v) if max_v is None else max_v
                mask = np.logical_and(v >= _min_v, v <= _max_v)
                self._mask_b *= mask

        # Minimum or maximum SNR. Here we allow the user to provide their own
        # value of the RMS if they want to override the value provided when
        # determining the surface.

        _saved_rms = self.rms
        self.rms = RMS or self.rms
        if min_SNR is not None or max_SNR is not None:
            if side in ['front', 'both']:
                SNR = self.SNR(side='front', masked=False)
                _min_SNR = np.nanmin(SNR) if min_SNR is None else min_SNR
                _max_SNR = np.nanmax(SNR) if max_SNR is None else max_SNR
                mask = np.logical_and(SNR >= _min_SNR, SNR <= _max_SNR)
                self._mask_f *= mask
            if side in ['back', 'both']:
                SNR = self.SNR(side='back', masked=False)
                _min_SNR = np.nanmin(SNR) if min_SNR is None else min_SNR
                _max_SNR = np.nanmax(SNR) if max_SNR is None else max_SNR
                mask = np.logical_and(SNR >= _min_SNR, SNR <= _max_SNR)
                self._mask_b *= mask
        self.rms = _saved_rms

    def _sigma_clip(self, p, side='front', reflect=True, masked=True,
                    nsigma=1.0, niter=3, window=0.1, min_sigma=0.0):
        """
        Apply a mask based on an iterative sigma clip for a given parameter.

        Args:
            p (str): The parameter to apply the sigma clipping to, for example
                ``'z'`` to clip based on the emission height.
            side (optional[str]): Side of the disk. Must be ``'front'``,
                ``'back'`` or ``'both'``. Defaults to ``'both'``.
            reflect (optional[bool]): Whether to reflect the backside points
                about the midplane. Defaults to ``False``.
            masked (optional[bool]): Whether to return only the masked points,
                the default, or all points.
            nsigma (optional[float]): The threshold for clipping in number of
                standard deviations.
            niter (optional[int]): The number of iterations to perform.
            window (optional[float]): The size of the window to use to
                calculate the standard deviation as a fraction of the beam
                FWHM.
            min_sigma (optional[float]): The minimum standard deviation
                possible as to avoid clipping all points.
        """
        raise NotImplementedError
        r = ', reflect={}'.format(str(reflect)) if p == 'z' else ''
        x = "self.{}(side='{}', masked={}{})".format(p, side, masked, r)
        for n in range(niter):
            x0 = self.rolling_statistic(p, func=np.nanmean, window=window,
                                        side=side, masked=masked)
            dx = self.rolling_statistic(p, func=np.nanstd, window=window,
                                        side=side, masked=masked)
            dx = np.clip(dx, a_min=min_sigma, a_max=None)
            mask = abs(eval(x) - x0) < nsigma * dx
            if side in ['front', 'both']:
                self._mask_f *= mask

    @staticmethod
    def convolve(x, N=7):
        """Convolve x with a Hanning kernel of size ``N``."""
        kernel = np.hanning(N)
        kernel /= kernel.sum()
        x_a = np.convolve(x, kernel, mode='same')
        x_b = np.convolve(x[::-1], kernel, mode='same')[::-1]
        return np.mean([x_a, x_b], axis=0)

    # -- BINNING FUNCTIONS -- #

    def binned_surface(self, rvals=None, rbins=None, side='front',
                       reflect=True, masked=True):
        """
        Bin the emisison surface onto a regular grid. This is a simple wrapper
        to the ``binned_parameter`` function.

        Args:
            rvals (optional[array]): Desired bin centers.
            rbins (optional[array]): Desired bin edges.
            side (optional[str]): Which 'side' of the disk to bin, must be one
                of ``'both'``', ``'front'`` or ``'back'``.
            reflect (Optional[bool]): Whether to reflect the emission height of
                the back side of the disk about the midplane.
            masked (Optional[bool]): Whether to use the masked data points.
                Default is ``True``.

        Returns:
            The bin centers, ``r``, and the average emission surface, ``z``,
            with the uncertainty, ``dz``, given as the bin standard deviation.
        """
        return self.binned_parameter('z', rvals=rvals, rbins=rbins, side=side,
                                     reflect=reflect, masked=masked)

    def binned_velocity_profile(self, rvals=None, rbins=None, side='front',
                                reflect=True, masked=True):
        """
        Bin the velocity onto a regular grid. This is a simple wrapper to the
        ``binned_parameter`` function.

        Args:
            rvals (optional[array]): Desired bin centers.
            rbins (optional[array]): Desired bin edges.
            side (optional[str]): Which 'side' of the disk to bin, must be one
                of ``'both'``', ``'front'`` or ``'back'``.
            reflect (Optional[bool]): Whether to reflect the emission height of
                the back side of the disk about the midplane.
            masked (Optional[bool]): Whether to use the masked data points.
                Default is ``True``.

        Returns:
            The bin centers, ``r``, and the average emission surface, ``z``,
            with the uncertainty, ``dz``, given as the bin standard deviation.
        """
        return self.binned_parameter('v', rvals=rvals, rbins=rbins, side=side,
                                     reflect=reflect, masked=masked)

    def binned_parameter(self, p, rvals=None, rbins=None, side='front',
                         reflect=True, masked=True):
        """
        Bin the provided parameter onto a regular grid. If neither ``rvals``
        nor ``rbins`` is specified, will default to 50 bins across the radial
        range of the bins.

        Args:
            p (str): Parameter to bin. For example, to bin the emission height,
                ``p='z'``.
            rvals (optional[array]): Desired bin centers.
            rbins (optional[array]): Desired bin edges.
            side (optional[str]): Which 'side' of the disk to bin, must be one
                of ``'both'``', ``'front'`` or ``'back'``.
            reflect (optional[bool]): Whether to reflect the emission height of
                the back side of the disk about the midplane.
            masked (optional[bool]): Whether to use the masked data points.
                Default is ``True``.

        Returns:
            The bin centers, ``r``, and the binned mean, ``mu``, and standard
            deviation, ``std``, of the desired parameter.
        """
        r = ', reflect={}'.format(str(reflect)) if p == 'z' else ''
        x = eval("self.{}(side='{}', masked={}{})".format(p, side, masked, r))
        rvals, rbins = self._get_bins(rvals=rvals, rbins=rbins, side=side,
                                      masked=masked)
        ridxs = np.digitize(self.r(side=side, masked=masked), rbins)
        avg = [np.nanmean(x[ridxs == rr]) for rr in range(1, rbins.size)]
        std = [np.nanstd(x[ridxs == rr]) for rr in range(1, rbins.size)]
        return rvals, np.squeeze(avg), np.squeeze(std)

    def _get_bins(self, rvals=None, rbins=None, side='front', masked=True):
        """Generate bins based on desired radial sampling."""
        if rvals is None and rbins is None:
            r = self.r(side=side, masked=masked)
            rbins = np.linspace(r.min(), r.max(), 51)
            rvals = 0.5 * (rbins[1:] + rbins[:-1])
        elif rvals is None:
            rvals = 0.5 * (rbins[1:] + rbins[:-1])
        elif rbins is None:
            rbins = 0.5 * np.diff(rvals).mean()
            rbins = np.linspace(rvals[0]-rbins, rvals[-1]+rbins, rvals.size+1)
        if not np.all(np.isclose(rvals, 0.5 * (rbins[1:] + rbins[:-1]))):
            print("Non-uniform bins detected - some functions may fail.")
        return rvals, rbins

    # -- ROLLING AVERAGE FUNCTIONS -- #

    def rolling_surface(self, window=0.1, side='front', reflect=True,
                        masked=True):
        """
        Return the rolling average of the emission surface. As the radial
        sampling is unevenly spaced the kernel size, which is a fixed number of
        samples, can vary in the radial range it represents. The uncertainty is
        taken as the rolling standard deviation.

        Args:
            window (optional[float]): Window size in [arcsec].
            side (optional[str]): Which 'side' of the disk to bin, must be one
                of ``'both'``', ``'front'`` or ``'back'``.
            reflect (optional[bool]): Whether to reflect the emission height of
                the back side of the disk about the midplane.
            masked (optional[bool]): Whether to use the masked data points.
                Default is ``True``.

        Returns:
            The radius, ``r``, emission height, ``z``, and uncertainty, ``dz``.
        """
        r, z = self.rolling_statistic(p='z', func=np.nanmean, window=window,
                                      side=side, reflect=reflect,
                                      masked=masked, remove_NaN=False)
        r, dz = self.rolling_statistic(p='z', func=np.nanstd, window=window,
                                       side=side, reflect=reflect,
                                       masked=masked, remove_NaN=False)
        idx = np.isfinite(z) & np.isfinite(dz)
        return np.squeeze(r[idx]), np.squeeze(z[idx]), np.squeeze(dz[idx])

    def rolling_velocity_profile(self, window=0.1, side='front', reflect=True,
                                 masked=True):
        """
        Return the rolling average of the velocity profile. As the radial
        sampling is unevenly spaced the kernel size, which is a fixed number of
        samples, can vary in the radial range it represents. The uncertainty is
        taken as the rolling standard deviation.

        Args:
            window (optional[float]): Window size in [arcsec].
            side (optional[str]): Which 'side' of the disk to bin, must be one
                of ``'both'``', ``'front'`` or ``'back'``.
            reflect (optional[bool]): Whether to reflect the emission height of
                the back side of the disk about the midplane.
            masked (optional[bool]): Whether to use the masked data points.
                Default is ``True``.

        Returns:
            The radius, ``r``, velocity, ``v``, and uncertainty, ``dv``.
        """
        r, v = self.rolling_statistic(p='v', func=np.nanmean, window=window,
                                      side=side, reflect=reflect,
                                      masked=masked, remove_NaN=False)
        r, dv = self.rolling_statistic(p='v', func=np.nanstd, window=window,
                                       side=side, reflect=reflect,
                                       masked=masked, remove_NaN=False)
        idx = np.isfinite(v) & np.isfinite(dv)
        return np.squeeze(r[idx]), np.squeeze(v[idx]), np.squeeze(dv[idx])

    def rolling_statistic(self, p, func=np.nanmean, window=0.1, side='front',
                          reflect=True, masked=True, remove_NaN=True):
        """
        Return the rolling statistic of the provided parameter.  As the radial
        sampling is unevenly spaced the kernel size, which is a fixed number of
        samples, can vary in the radial range it represents.

        Args:
            p (str): Parameter to apply the rolling statistic to. For example,
                to use the emission height, ``p='z'``.
            func (Optional[callable]): The function to apply to the data.
            window (Optional[float]): Window size in [arcsec].
            side (Optional[str]): Which 'side' of the disk to bin, must be one
                of ``'both'``', ``'front'`` or ``'back'``.
            reflect (Optional[bool]): Whether to reflect the emission height of
                the back side of the disk about the midplane.
            masked (Optional[bool]): Whether to use the masked data points.
                Default is ``True``.
            remove_NaN (Optional[bool]): Whether to remove the NaNs.

        Returns:
            The radius, ``r`` and the rolling statistic, ``s``. All NaNs will
            have been removed.
        """
        r = ', reflect={}'.format(str(reflect)) if p == 'z' else ''
        x = self.r(side=side, masked=masked)
        y = eval("self.{}(side='{}', masked={}{})".format(p, side, masked, r))
        idx = np.argsort(x)
        x, y = x[idx], y[idx]
        w = self._get_rolling_stats_window(window=window,
                                           masked=masked,
                                           side=side)
        e = int((w - 1) / 2)
        yy = np.insert(y, 0, y[0] * np.ones(e))
        yy = np.insert(yy, -1, y[-1] * np.ones(e))
        s = np.squeeze([func(y[i-e+1:i+e+2]) for i in range(y.size)])
        if remove_NaN:
            idx = np.isfinite(s)
            return x[idx], s[idx]
        else:
            return x, s

    def _get_rolling_stats_window(self, window=0.1, side='front', masked=True):
        """Size of the window used for rolling statistics."""
        dr = np.diff(self.r(side=side, masked=masked))
        dr = np.where(dr == 0.0, 1e-10, dr)
        w = np.median(window / dr).astype('int')
        return w if w % 2 else w + 1

    # -- INTERPOLATION FUNCTIONS -- #

    def interpolate_parameter(self, p, method='rolling', smooth=7,
                              interp1d_kw=None, func=np.nanmean, window=0.1,
                              remove_NaN=True, rvals=None, rbins=None,
                              side='front', reflect=True, masked=True):
        """
        Return an interpolatable function for a given parameter. This function
        is essentially a wrapper for ``scipy.interpolate.interp1d``.

        Args:
            p (str): Parameter to return an interpolation of.
            method (optional[str]): Method used to create an initial radial
                profile of the parameter, either a rolling statistic with
                ``'rolling'`` or a radially binned statistic with ``'binned'``.
            smooth (optional[int]): Smooth the profile by convolving with a
                Hanning kernel with a size of ``smooth``.
            interp1d_kw (optional[dict]): Kwargs to pass to
                ``scipy.interpolate.interp1d``.
            func (Optional[callable]): The function to apply to the data if
                using ``method='rolling'``.
            window (Optional[float]): Window size in [arcsec] to use if using
                ``method='rolling'``.
            remove_NaN (Optional[bool]): Whether to remove the NaNs if using
                ``method='rolling'``.
            rvals (optional[array]): Desired bin centers if using
                ``method='binned'``.
            rbins (optional[array]): Desired bin edges if using
                ``method='binned'``.
            side (Optional[str]): Which 'side' of the disk to bin, must be one
                of ``'both'``', ``'front'`` or ``'back'``.
            reflect (Optional[bool]): Whether to reflect the emission height of
                the back side of the disk about the midplane.
            masked (Optional[bool]): Whether to use the masked data points.
                Default is ``True``.

        Returns:
            An ```interp1d`` instance of the (optionally smoothed) radial
            profile.
        """

        # Grab the radial profile.

        if method == 'rolling':
            x, y = self.rolling_statistic(p, func=func, window=window,
                                          side=side, reflect=reflect,
                                          masked=masked,
                                          remove_NaN=remove_NaN)
        elif method == 'binned':
            x, y, _ = self.binned_parameter(p, rvals=rvals, rbins=rbins,
                                            side=side, reflect=reflect,
                                            masked=masked)
        else:
            raise ValueError("`method` must be either 'rolling' or 'binned'.")

        # Smooth the radial profile if necessary.

        if smooth:
            y = self.convolve(y, smooth)

        # Build the interpolation function and return.

        from scipy.interpolate import interp1d
        interp1d_kw = {} if interp1d_kw is None else interp1d_kw
        interp1d_kw['bounds_error'] = interp1d_kw.pop('bounds_error', False)
        interp1d_kw['fill_value'] = interp1d_kw.pop('fill_value', np.nan)
        return interp1d(x, y, **interp1d_kw)

    # -- FITTING FUNCTIONS -- #

    def fit_emission_surface(self, tapered_powerlaw=True, include_cavity=False,
                             r0=None, dist=None, side='front', masked=True,
                             return_model=False, curve_fit_kwargs=None):
        r"""
        Fit the extracted emission surface with a tapered power law of the form

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
        the covariance maxtrix returned by ``curve_fit``. We use the SNR of
        each point as a weighting in the fit.

        Args:
            tapered_powerlaw (optional[bool]): If ``True``, fit the tapered
                power law profile rather than a single power law function.
            include_cavity (optional[bool]): If ``True``, include a cavity in
                the functional form, inside of which all heights are set to 0.
            r0 (optional[float]): The reference radius for :math:`z_0`.
                Defaults to 1 arcsec, unless ``dist`` is provided, then
                defaults to 100 au.
            dist (optional[float]): Convert all distances from [arcsec] to [au]
                for the fitting. If this is provided, ``r_ref`` will change to
                100 au unless specified by the user.
            side (optional[str]): Which 'side' of the disk to bin, must be one
                of ``'both'``', ``'front'`` or ``'back'``.
            masked (optional[bool]): Whether to use the masked data points.
                Default is ``True``.
            curve_fit_kwargs (optional[dict]): Keyword arguments to pass to
                ``scipy.optimize.curve_fit``.

        Returns:
            Best-fit values, ``popt``, and associated uncertainties, ``copt``,
            for the fits if ``return_fit=False``, else the best-fit model
            evaluated at the radial points.
        """
        from scipy.optimize import curve_fit

        if side.lower() not in ['front', 'back', 'both']:
            raise ValueError(f"Unknown `side` values {side}.")
        r = self.r(side=side, masked=masked)
        z = self.z(side=side, reflect=True, masked=masked)
        dz = 1.0 / self.SNR(side=side, masked=masked)
        nan_mask = np.isfinite(r) & np.isfinite(z) & np.isfinite(dz)
        r, z, dz = r[nan_mask], z[nan_mask], dz[nan_mask]
        idx = np.argsort(r)
        r, z, dz = r[idx], z[idx], dz[idx]

        # If a distance is provided, convert all distances to [au]. We also
        # change the reference radius to 100 au unless specified.

        if dist is not None:
            r, z, dz = r * dist, z * dist, dz * dist
            r0 = r0 or 100.0
        else:
            dist = 1.0
            r0 = r0 or 1.0

        kw = {} if curve_fit_kwargs is None else curve_fit_kwargs
        kw['maxfev'] = kw.pop('maxfev', 100000)
        kw['sigma'] = dz

        kw['p0'] = [0.3 * dist, 1.0]
        if tapered_powerlaw:
            def func(r, *args):
                return surface._tapered_powerlaw(r, *args, r0=r0)
            kw['p0'] += [1.0 * dist, 1.0]
        else:
            def func(r, *args):
                return surface._powerlaw(r, *args, r0=r0)
        if include_cavity:
            kw['p0'] += [0.05 * dist]

        try:
            popt, copt = curve_fit(func, r, z, **kw)
            copt = np.diag(copt)**0.5
        except RuntimeError:
            popt = kw['p0']
            copt = [np.nan for _ in popt]

        return (r, func(r, *popt)) if return_model else (popt, copt)

    def fit_emission_surface_MCMC(self, r0=None, dist=None, side='front',
                                  masked=True, tapered_powerlaw=True,
                                  include_cavity=False, p0=None, nwalkers=64,
                                  nburnin=1000, nsteps=500, scatter=1e-3,
                                  priors=None, returns=None, plots=None,
                                  curve_fit_kwargs=None, niter=1):
        r"""
        Fit the inferred emission surface with a tapered power law of the form

        .. math::
            z(r) = z_0 \, \left( \frac{r}{r_0} \right)^{\psi}
            \times \exp \left( -\left[ \frac{r}{r_{\rm taper}}
            \right]^{q_{\rm taper}} \right)

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
            r0 (Optional[float]): The reference radius for :math:`z_0`.
                Defaults to 1 arcsec, unless ``dist`` is provided, then
                defaults to 100 au.
            dist (Optional[float]): Convert all distances from [arcsec] to [au]
                for the fitting. If this is provided, ``r_ref`` will change to
                100 au unless specified by the user.
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

        r = self.r(side=side, masked=masked)
        z = self.z(side=side, reflect=True, masked=masked)
        dz = 1.0 / self.SNR(side=side, masked=masked)
        nan_mask = np.isfinite(r) & np.isfinite(z) & np.isfinite(dz)
        r, z, dz = r[nan_mask], z[nan_mask], dz[nan_mask]
        idx = np.argsort(r)
        r, z, dz = r[idx], z[idx], dz[idx]

        # If a distance is provided, convert all distances to [au]. We also
        # change the reference radius to 100 au unless specified.

        if dist is not None:
            r, z, dz = r * dist, z * dist, dz * dist
            r0 = r0 or 100.0
        else:
            dist = 1.0
            r0 = r0 or 1.0

        # Set the initial guess if not provided.

        if p0 is None:
            p0 = [0.3 * dist, 1.0]
            if tapered_powerlaw:
                p0 += [1.0 * dist, 1.0]
            if include_cavity:
                p0 += [0.05 * dist]

        # Default number of walkers to twice the number of free parameters.

        nwalkers = max(nwalkers, 2 * len(p0))

        # Define the labels.

        labels = ['z0', 'psi']
        if tapered_powerlaw:
            labels += ['r_taper', 'q_taper']
        if include_cavity:
            labels += ['r_cavity']
        assert len(labels) == len(p0)

        # Set the priors for the MCMC.

        priors = {} if priors is None else priors
        priors['z0'] = priors.pop('z0', [0.0, 5.0 * dist, 'flat'])
        priors['psi'] = priors.pop('psi', [0.0, 5.0, 'flat'])
        priors['r_taper'] = priors.pop('r_taper', [0.0, 2 * r.max(), 'flat'])
        priors['q_taper'] = priors.pop('q_taper', [0.0, 5.0, 'flat'])
        priors['r_cavity'] = priors.pop('r_cavity', [0.0, r.max() / 2, 'flat'])

        # Set the starting positions for the walkers.

        for _ in range(niter):
            p0 = surface._random_p0(p0, scatter, nwalkers)
            args = (r, z, dz, labels, priors, r0)
            sampler = emcee.EnsembleSampler(nwalkers, p0.shape[1],
                                            self._ln_probability,
                                            args=args)
            sampler.run_mcmc(p0, nburnin + nsteps, progress=True)
            samples = sampler.chain[:, -int(nsteps):]
            samples = samples.reshape(-1, samples.shape[-1])
            p0 = np.median(samples, axis=0)
        walkers = sampler.chain.T

        # Diagnostic plots.

        plots = ['corner'] if plots is None else np.atleast_1d(plots)
        if 'walkers' in plots:
            surface._plot_walkers(walkers, labels, nburnin)
        if 'corner' in plots:
            surface._plot_corner(samples, labels)

        # Generate the output.

        to_return = []
        for tr in ['median'] if returns is None else np.atleast_1d(returns):
            if tr == 'walkers':
                to_return += [walkers]
            if tr == 'samples':
                to_return += [samples]
            if tr == 'lnprob':
                to_return += [sampler.lnprobability[nburnin:]]
            if tr == 'percentiles':
                to_return += [np.percentile(samples, [16, 50, 84], axis=0).T]
            if tr == 'median':
                to_return += [np.median(samples, axis=0)]
            if tr == 'model':
                median = np.median(samples, axis=0)
                to_return += [r, surface._parse_model(r, median, labels, r0)]
        return to_return if len(to_return) > 1 else to_return[0]

    def _ln_probability(self, theta, r, z, dz, labels, priors, r0):
        """Log-probabiliy function for the emission surface fitting."""
        lnp = 0.0
        for label, t in zip(labels, theta):
            lnp += surface._ln_prior(priors[label], t)
        if not np.isfinite(lnp):
            return lnp
        model = surface._parse_model(r, theta, labels, r0)
        lnx2 = -0.5 * np.sum(np.power((z - model) / dz, 2)) + lnp
        return lnx2 if np.isfinite(lnx2) else -np.inf

    @staticmethod
    def _parse_model(r, theta, labels, r0):
        """Parse the model parameters."""
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
        return surface._tapered_powerlaw(r=r, z0=z0, q=q, r_taper=r_taper,
                                         q_taper=q_taper, r_cavity=r_cavity,
                                         r0=r0)

    @staticmethod
    def _powerlaw(r, z0, q, r_cavity=0.0, r0=1.0):
        """Standard power law profile."""
        rr = np.clip(r - r_cavity, a_min=0.0, a_max=None)
        return z0 * (rr / r0)**q

    @staticmethod
    def _tapered_powerlaw(r, z0, q, r_taper=np.inf, q_taper=1.0, r_cavity=0.0,
                          r0=1.0):
        """Exponentially tapered power law profile."""
        rr = np.clip(r - r_cavity, a_min=0.0, a_max=None)
        f = surface._powerlaw(rr, z0, q, r_cavity=0.0, r0=r0)
        return f * np.exp(-(rr / r_taper)**q_taper)

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

    # -- PLOTTING FUNCTIONS -- #

    def plot_surface(self, ax=None, side='both', reflect=False, masked=True,
                     plot_fit=False, tapered_powerlaw=True,
                     include_cavity=False, return_fig=False):
        """
        Plot the emission surface.

        Args:
            ax (Optional[Matplotlib axis]): Axes used for plotting.
            masked (Optional[bool]): Whether to plot the maske data or not.
                Default is ``True``.
            side (Optional[str]): Which emission side to plot, must be
                ``'front'``, ``'back'`` or ``'both'``.
            reflect (Optional[bool]): If plotting the ``'back'`` side of the
                disk, whether to reflect it about disk midplane.
            tapered_powerlaw (Optional[bool]): TBD
            include_cavity (Optional[bool]): TBD
            return_fig (Optional[bool]): Whether to return the Matplotlib
                figure if ``ax=None``.

        Returns:
            If ``return_fig=True``, the Matplotlib figure used for plotting.
        """

        # Generate plotting axes.

        if ax is None:
            fig, ax = plt.subplots()
        else:
            return_fig = False

        # Plot each side separately to have different colors.

        if side.lower() not in ['front', 'back', 'both']:
            raise ValueError(f"Unknown `side` value {side}.")
        if side.lower() in ['back', 'both']:
            r = self.r(side='back', masked=masked)
            z = self.z(side='back', reflect=reflect, masked=masked)
            ax.scatter(r, z, color='r', marker='.', alpha=0.2)
            ax.scatter(np.nan, np.nan, color='r', marker='.', label='back')
        if side.lower() in ['front', 'both']:
            r = self.r(side='front', masked=masked)
            z = self.z(side='front', masked=masked)
            ax.scatter(r, z, color='b', marker='.', alpha=0.2)
            ax.scatter(np.nan, np.nan, color='b', marker='.', label='front')

        # Plot the fit.

        if plot_fit:
            r, z = self.fit_emission_surface(tapered_powerlaw=tapered_powerlaw,
                                             include_cavity=include_cavity,
                                             side=side, masked=masked,
                                             return_model=True)
            idx = np.argsort(r)
            r, z = r[idx], z[idx]
            if side in ['front', 'both']:
                ax.plot(r, z, color='k', lw=1.5)
            if side in ['back', 'both'] and not reflect:
                ax.plot(r, -z, color='k', lw=1.5)

        # Gentrification.

        ax.set_xlabel("Radius (arcsec)")
        ax.set_ylabel("Height (arcsec)")
        ax.legend(markerfirst=False)

        # Returns.

        if return_fig:
            return fig

    def plot_velocity_profile(self, ax=None, plot_rolling=False, masked=True,
                              return_fig=False, window=0.1):
        """
        Plot the measured velocity profile.

        Args:
            ax (Optional[Matplotlib axis]): Axes used for plotting.
            masked (Optional[bool]): Whether to plot the maske data or not.
                Default is ``True``.
            return_fig (Optional[bool]): Whether to return the Matplotlib
                figure if ``ax=None``.

        Returns:
            If ``return_fig=True``, the Matplotlib figure used for plotting.
        """

        # Generate plotting axes.

        if ax is None:
            fig, ax = plt.subplots()
        else:
            return_fig = False

        # Plot the velocity profiles.

        r = self.r(side='front', masked=masked)
        v = self.v(side='front', masked=masked)
        ax.scatter(r, v, color='k', marker='.', alpha=0.2)

        if plot_rolling:
            x, y, dy = self.rolling_velocity_profile(window=window,
                                                     side='front',
                                                     masked=masked)
            ax.fill_between(x, y - dy, y + dy, color='r', lw=0.0, alpha=0.2)
            ax.plot(x, y, color='r', lw=1.0, label='rolling mean')

        # Gentrification.

        ax.set_xlabel("Radius (arcsec)")
        ax.set_ylabel(r"v$_{\phi}$ (arcsec)")
        ax.legend(markerfirst=False)

        # Returns.

        if return_fig:
            return fig

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
        """Plot the walker traces."""
        import matplotlib.pyplot as plt
        for param, label in zip(walkers, labels):
            fig, ax = plt.subplots()
            for walker in param.T:
                ax.plot(walker, alpha=0.1)
            ax.axvline(nburnin)
            ax.set_ylabel(label)
            ax.set_xlabel('Steps')
