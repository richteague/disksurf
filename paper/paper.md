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
  - name: Feilong Meng
    orcid: 0000-0003-0079-6830
    affiliation: 2
affiliations:
 - name: Center for Astrophysics | Harvard & Smithsonian, 60 Garden St., Cambridge, MA 02138, USA
   index: 1
 - name: Department of Astronomy, University of Michigan, 323 West Hall, 1085 South University Avenue, Ann Arbor, MI 48109, USA
   index: 2
 - name: NASA Hubble Fellowship Program Sagan Fellow
   index: 3
date: 29 September 2021
bibliography: paper.bib

---

# Summary

 `disksurf` implements the method presented in @Pinte:2018 to extract the molecular emission surface (i.e., the height above the midplane from which molecular emission arises) in moderately inclined protoplanetary disks. The Python-based code leverages the open-source `GoFish` [@GoFish] package to read in and interact with FITS data cubes used for essentially all astronomical observations at submillimeter wavelengths. The code also uses the open-source `detect_peaks.py` routine from @Duarte:2021 for peak detection. For a given set of geometric parameters specified by the user, `disksurf` will return a surface object containing both the disk-centric coordinates of the surface as well as the gas temperature and rotation velocity at those locations. The user is able to 'filter' the returned surface using a variety of clipping and smoothing functions. Several simple analytical forms commonly adopted in the protoplanetary disk literature can then be fit to this surface using either a chi-squared minimization with `scipy` [@Virtanen:2020] or through an Monte-Carlo Markov-Chain approach with `emcee` [@Foreman-Mackey:2013]. To verify the 3D geometry of the system is well constrained, `disksurf` also provides diagnostic functions to plot the emission surface over channel maps of line emission (i.e., the emission morphology for a specific frequency).

# Statement of need

The Atacama Millimeter/submillimeter Array (ALMA) has brought our view of protoplanetary disks, the formation environment of planets, into sharp focus. The unparalleled angular resolution now achievable with ALMA allows us to routinely resolve the 3D structure of these disks; detailing the vertical structure of the gas and dust from which planets are formed. Extracting the precise height from where emission arises is a key step towards understanding the conditions in which a planet is born, and, in turn, how the planet can affect the parental disk.

A method for extracting a 'scattering surface', the emission surface equivalent for small, submicron grains was described in @Stolker:2016 who provided the `diskmap` package. However, this approach is not suitable for molecular emission, which traces the gas component of the disk and has a strong frequency dependence due to Doppler shifting from the disk rotation. @Pinte:2018 presented an alternative method that could account for this frequency dependence and demonstrated that this could be used to trace key physical properties of the protoplanetary disk, namely the gas temperature and rotation velocity, along the emission surface.

While the measurement of the emission surface only requires simple geometrical transformations, the largest source of uncertainty arises through the handling of noisy data. As more works perform such analyses, for example @Teague:2019, @Rich:2021, and @Law:2021, the need for an easy-to-use package that implements this method was clear. Such a package would facilitate the rapid reproduction of published results, enable direct comparisons between numerical simulations and observations [@Calahan:2021; @Schwarz:2021], and ease benchmarking between different publications. `disksurf` provides this functionality, along with a tutorial to guide users through the process of extracting an emission surface. The code is developed in such a way that as the quality of observations improve, the extraction methods can be easily refined to maintain precise measurements of the emission surface.

# Acknowledgements

We acknowledge help from Christophe Pinte in benchmarking early versions of the code with those presented in the original paper detailing the method, @Pinte:2018. R.T. acknowledges support from the Smithsonian Institution as a Submillimeter Array (SMA) Fellow. C.J.L. acknowledges funding from the National Science Foundation Graduate Research Fellowship under Grant DGE1745303. Support for J.H. was provided by NASA through the NASA Hubble Fellowship grant #HST-HF2-51460.001- A awarded by the Space Telescope Science Institute, which is operated by the Association of Universities for Research in Astronomy, Inc., for NASA, under contract NAS5-26555.

# References
