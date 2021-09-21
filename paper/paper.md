---
title: 'disksurf: Extracting the 3D Structure of Protoplanetary Disks'
tags:
  - Python
  - astronomy
  - protoplanetary disks
authors:
  - name: Richard Teague
    orcid: 0000-0003-0872-7098
    affiliation: 1
  - name: Charles J. Law
    orcid: 0000-0003-1413-1776
    affiliation: 1
  - name: Jane Huang
    orcid: 0000-0001-6947-6072
    affiliation: "2, 3"
  - name:
    affiliation: 2
affiliations:
 - name: Center for Astrophysics | Harvard & Smithsonian, 60 Garden St., Cambridge, MA 02138, USA
   index: 1
 - name: Department of Astronomy, University of Michigan, 323 West Hall, 1085 South University Avenue, Ann Arbor, MI 48109, USA
   index: 2
 - name: NASA Hubble Fellowship Program Sagan Fellow
   index: 3
date: 20 September 2021
bibliography: paper.bib

---

# Summary

 `disksurf` implements the method presented in `@pinte:2018` to extract the molecular emission surface (height above the midplane from which molecular emission arises) in moderately inclined protoplanetary disks. The Python based code leverages the `gofish` `[@gofish]` package to read in and interact with FITS data cubes used for essentially all sub-mm observations. For a given set of geometric parameters specified by the user, `disksurf` will return a surface object containing both the disk-centric coordinates of the surface as well as the gas temperature and rotation velocity at those locations. The user is able to 'clean' the returned surface using a variety of clipping and smoothing functions. Several simple analytical forms commonly adopted in the protoplanetary disk literature can then be fit to this surface using either a chi-squared minimization with `scipy` or through an MCMC approach with `emcee` `[@Foreman-Mackey:2016]`. To verify the 3D geometry of the system is well constrained, `disksurf` also provides convenience functions to plot the emission surface over the channel maps.

# Statement of need

The Atacama Millimeter/submillimeter Array has brought our view of protoplanetary disks, the formation environment of planets, into sharp focus. The unparalleled angular resolution now achievable allows us to routinely resolve the 3D structure of these disks; detailing the vertical structure of the gas and dust from which planets are formed. Extracting the precise height from where emission arises is a key step towards understanding the conditions in which a planet is born, and, in turn, how the planet can affect the parental disk.

A method for extracting a 'scattering surface', the emission surface equivalent for small, sub-micron grains was described in `@stolker:2016` who provided the `diskmap` package. However, as molecular emission, which traces the gas component of the disk, has a strong frequency dependence due the rotation of the disk Doppler shifting the emission, this approach is unfeasible. `@pinte:2018` presented an alternative method that could account for this frequency dependence, and demonstrated that this could be used to trace key physical properties of the protoplanetary disk, namely the gas temperature and rotation velocity, along the emission surface.

While the measurement of the emission surface only requires simple geometrical transforms, the largest source of uncertainty arises through the handling of noisy data. As more works perform such analyses, for example `@Teague:2019`, `@Rich:2021` and `@Law:2021`, the need for an easy-to-use package that implements this method was clear. Such a package would facilitate the rapid reproduction of published results and ease direct comparisons between different publications. `disksurf` provides this functionality, along with a tutorial to guide users through the process of extracting an emission surface. The code is developed in such a way that as the quality of observations improve, the extraction methods can be easily refined to maintain precise measurements of the emission surface.

# Acknowledgements

We acknowledge help from Christophe Pinte in benchmarking early versions of the
code with those presented in the original paper detailing the method, `@pinte:2018`.

# References
