Methodology
***********

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


Constructing the CDF \& PDF
===========================

In this notebook, we describe how NumDF approximates the CDF
and PDF of a function using firedrake. We describe the choice
of function spaces used and details of the slope limiter required
to gaurantee a monotonic and right increasing CDF.  A rendered
version of this notebook is available `here
<https://nbviewer.org/github/mannixp/D.stratify-pdfe/blob/main/notebooks/explanatory_notebooks/Part1_CDF_and_PDF_Construction.ipynb>`__
and there is a version `on Colab <https://colab.research.google.com/github/mannixp/D.stratify-pdfe/blob/main/notebooks/explanatory_notebooks/Part1_CDF_and_PDF_Construction.ipynb>`__


Constructing the inverse CDF or QDF
===================================

Next, we discuss how to compute the inverse CDF also known 
as the QDF or quantile density function `here <https://nbviewer.org/github/mannixp/D.stratify-pdfe/blob/main/notebooks/explanatory_notebooks/Part2_QDF_Construction.ipynb>`__.
You can run this notebook yourself `on Colab <https://colab.research.google.com/github/mannixp/D.stratify-pdfe/blob/main/notebooks/explanatory_notebooks/Part2_QDF_Construction.ipynb>`__


Composing density objects
=========================

Finally we discuss how integrals of compositions of the CDF and PDF can be evaluated
`here <https://nbviewer.org/github/mannixp/D.stratify-pdfe/blob/main/notebooks/explanatory_notebooks/Part3_APE_Calculation.ipynb>`__.
You can run this notebook yourself `on Colab
<https://colab.research.google.com/github/mannixp/D.stratify-pdfe/blob/main/notebooks/explanatory_notebooks/Part3_APE_Calculation.ipynb>`__

