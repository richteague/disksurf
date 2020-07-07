# diskprojection

<p align='center'>
  <img src="HD163296_zeroth.png" width="793" height="549">
</p>

## What is it?

Functions to measure the height of optically thick emission using the method presented in [Pinte et al. (2018a)](https://ui.adsabs.harvard.edu/abs/2018A%26A...609A..47P/abstract), then use this information to deproject images into arbitrary angles. Very much a work in progress.

## How do I use it?

```python
from diskprojection import disk_observation

# load up the observations
disk = disk_observation('path/to/imagecube.fits')

# grab the emission surface
r, z, Fnu, v = disk.get_emission_surface(inc=30.0, PA=35.0)

# make some cuts to get rid of poor points
r, z, Fnu, v = disk.clip_emission_surface(r, z, Fnu, v, min_Fnu=0.05)

# apply an iterative sigma clipping to remove noise
r, z, Fnu, v = disk.iterative_clip_emission_surface(r, z, Fnu, v)
```

#### Notes

Part of the code uses `detect_peaks.py` from [Marcos Duarte](https://github.com/demotu/BMC). For this to run you need the [GoFish](https://github.com/richteague/gofish) package.
