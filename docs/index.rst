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


Citation
********

If you use this software, please cite both the Pinte et al. (2018) for the
methodology and the JOSS paper for this ``disksurf`` software.

.. code-block::

    @article{2018A&A...609A..47P,
      doi = {10.1051/0004-6361/201731377},
      year = {2018},
      volume = {609},
      eid = {A47},
      pages = {A47},
      author = {{Pinte}, C. and {M{\'e}nard}, F. and {Duch{\^e}ne}, G. and {Hill}, T. and {Dent}, W.~R.~F. and {Woitke}, P. and {Maret}, S. and {van der Plas}, G. and {Hales}, A. and {Kamp}, I. and {Thi}, W.~F. and {de Gregorio-Monsalvo}, I. and {Rab}, C. and {Quanz}, S.~P. and {Avenhaus}, H. and {Carmona}, A. and {Casassus}, S.},
      title = "{Direct mapping of the temperature and velocity gradients in discs. Imaging the vertical CO snow line around IM Lupi}",
      journal = {\aap}
    }

    @article{disksurf,
      doi = {10.21105/joss.03827},
      url = {https://doi.org/10.21105/joss.03827},
      year = {2021},
      publisher = {The Open Journal},
      volume = {6},
      number = {67},
      pages = {3827},
      author = {Richard Teague and Charles J. Law and Jane Huang and Feilong Meng},
      title = {disksurf: Extracting the 3D Structure of Protoplanetary Disks},
      journal = {Journal of Open Source Software}
    }


Support
*******

Information for all the functions can be found in the `API <https://disksurf.readthedocs.io/en/latest/user/api.html>`_ documentation.
If you are having issues, please open a `issue <https://github.com/richteague/disksurf/issues>`_ on the GitHub page.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   tutorials/tutorial_1
   user/api
