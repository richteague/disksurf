# disksurf

<p align='center'>
  <img src="HD163296_zeroth.png" width="793" height="549">
  <br>
  <a href='https://disksurf.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/disksurf/badge/?version=latest' alt='Documentation Status' />
  </a>
</p>

## What is it?

`disksurf` is a package which contains the functions to measure the height of optically thick emission, or photosphere, using the method presented in [Pinte et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018A%26A...609A..47P/abstract).

## How do I install it?

Grab the latest version from PyPI:

```
$ pip install disksurf
```

This has a couple of dependencies, namely [astropy](https://github.com/astropy/astropy) and [GoFish](https://github.com/richteague/gofish), which should be installed automatically if you don't have them. To verify that everything was installed as it should, running through the [tutorials](https://disksurf.readthedocs.io/en/latest/tutorials/tutorial_1.html) should work without issue.

## How do I use it?

At its most basic, `disksurf` is as easy as:

```python
from disksurf import observation                        # import the module
cube = observations('path/to/cube.fits')                # load up the data
surface = cube.get_emission_surface(inc=30.0, PA=45.0)  # extract the surface
surface.plot_surface()                                  # plot the surface
```

Follow our [tutorials](https://disksurf.readthedocs.io/en/latest/tutorials/tutorial_1.html) for a quick guide on how to use `disksurf` with DSHARP data and some of the additional functions that will help you extract the best surface possible.
