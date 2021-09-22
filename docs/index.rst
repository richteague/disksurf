disksurf
########

``disksurf`` is a package that implements the method described in
`Pinte et al. (2018) <https://ui.adsabs.harvard.edu/abs/2018A%26A...609A..47P/abstract>`_
to extract the emission surface from spatially and spectrally resolved
observations of molecular emission from a protoplanetary disk. The package
provides a suite of convenience functions to not only extract an emission
surface, but also fit commonly used analytical forms, and over plot isovelocity
contours on channel maps to verify the correct surface was extracted.

Installation
************

To install ``disksurf`` we'd recommend using PyPI:

.. code-block::

    pip install disksurf

Alternatively you can clone the repository and install that version.

.. code-block::

    git clone https://github.com/richteague/disksurf.git
    cd disksurf
    pip install .

To guide you through how to use ``disksurf`` we've created a
`tutorial <https://disksurf.readthedocs.io/en/latest/tutorials/tutorial_1.html>`_
using data from the `DSHARP <https://almascience.eso.org/almadata/lp/DSHARP/>`_
Large Program. This tutorial also serves as a test that the software has been
installed correctly.

Support
*******

Information for all the functions can be found in the `API <https://disksurf.readthedocs.io/en/latest/user/api.html>`_ documentation.
If you are having issues, please open a `issue <https://github.com/richteague/disksurf/issues>`_ on the GitHub page.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   tutorials/tutorial_1
   user/api
