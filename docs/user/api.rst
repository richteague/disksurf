API
===

`observation`
-------------

This class is built upon the `imagecube` class from `GoFish <https://github.com/richteague/gofish>`_,
so contains all functionality described there.

.. autoclass:: disksurf.observation
    :inherited-members:

`surface`
---------

The `surface` class is returned from the `get_emission_surface()` function and
was not designed to be created by a user (hence the rather long list of variables
required to instantiate the class).

.. autoclass:: disksurf.surface
    :inherited-members:
