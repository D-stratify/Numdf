Tutorials \& Examples
*********************

The following notebooks provide an overview of the methodology behind 
how NumDF constructs the CDF, PDF and QDF and computes compositions
of these objects. To run the notebooks, you will need to `install jupyter
<https://jupyter.org/install.html>`__ *inside* your activated
Firedrake virtualenv.

These notebooks are maintained in the NumDF repository, so all the
material is available in your NumDF installation source
directory while the notebooks are in the directory ``NumDF/docs/notebooks``.

Thanks to the excellent `FEM on
Colab <https://fem-on-colab.github.io/index.html>`__ by `Francesco
Ballarin <https://www.francescoballarin.it>`__, you can run the notebooks on
Google Colab through your web browser, without installing Firedrake or NumDF.

We also provide links to non-interactive renderings of the notebooks using
`Jupyter nbviewer <https://nbviewer.jupyter.org>`__.


Analytic functions
==================

In this notebook, we the basic functionality of NumDF including how to compute the
CDF, PDF and QDF of a one and two dimensional analytical function. A rendered version of this notebook is available `here
<https://nbviewer.org/github/mannixp/D.stratify-pdfe/blob/main/notebooks/example_notebooks/analytical_functions.ipynb>`__
and there is a version `on Colab <https://colab.research.google.com/github/mannixp/D.stratify-pdfe/blob/main/notebooks/example_notebooks/analytical_functions.ipynb>`__


Available potential energy
==========================

Next, we discuss how to compute the available potential energy for a simple two dimensional field. 
This example builds on the previous example by requiring the integral of the composition of two CDFs to be evaluated. 
A rendered version of this notebook is available `here
<https://nbviewer.org/github/mannixp/D.stratify-pdfe/blob/main/notebooks/example_notebooks/ape_calculation.ipynb>`__
and there is a version `on Colab <https://colab.research.google.com/github/mannixp/D.stratify-pdfe/blob/main/notebooks/example_notebooks/ape_calculation.ipynb>`__


Numerical functions
===================

We then consider a more practically relevant example where the function specified is obtained as the output of a direct numerical simulation. 
Considering a two dimensional Kelvin-Helmholtz instability we present the time evolution of the CDF for which a rendered version of this notebook is available `here
<https://nbviewer.org/github/mannixp/D.stratify-pdfe/blob/main/notebooks/example_notebooks/numerical_functions.ipynb>`__
and there is a version `on Colab <https://colab.research.google.com/github/mannixp/D.stratify-pdfe/blob/main/notebooks/example_notebooks/numerical_functions.ipynb>`__


Numerical Convergence
====================

Finally we show that the numerical implementation of our numerical method is consistent and discuss the challenges that arise when 
computing the density of functions. A rendered version of this notebook is available `here
<https://nbviewer.org/github/mannixp/D.stratify-pdfe/blob/main/notebooks/example_notebooks/convergence.ipynb>`__
and there is a version `on Colab <https://colab.research.google.com/github/mannixp/D.stratify-pdfe/blob/main/notebooks/example_notebooks/convergence.ipynb>`__
