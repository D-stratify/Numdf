"""Tests for the Ptp."""

from firedrake import *
from numdf import Ptp
import numpy as np


def test_initialise():
    """Ensure we can import and initialise Ptp."""
    # 1D
    Ptp(Omega_X={'x1': (-1, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=2)

    # 2D
    Ptp(Omega_X={'x1': (-1, 1), 'x2': (-1, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=2)

    assert True


def test_cdf_constant():
    """Check the CDF of Y(X)=0 is correctly calculated."""
    ptp = Ptp(Omega_X={'x1': (0, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=2)
    x1 = ptp.x_coords()
    
    density = ptp.fit(Y=0*x1)
    assert assemble(((density.cdf-1)**2)*dx) < 1e-8


def test_pdf_constant():
    """Check the PDF of Y(X)=0 is correctly calculated."""
    ptp = Ptp(Omega_X={'x1': (0, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=2)
    x1 = ptp.x_coords()
    density = ptp.fit(Y=0*x1)
    integral = density.distribution(1)

    assert abs(integral - 1) < 1e-12


def test_cdf_uniform_domain_length():
    """Check the CDF of Y(x1)=x1 is y."""
    ptp = Ptp(Omega_X={'x1': (0, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=5)
    x1 = ptp.x_coords()

    density = ptp.fit(Y=x1, quadrature_degree=1000)
    assert assemble(((density.cdf-density.y)**2)*dx) < 1e-8


def test_pdf_uniform_domain_length():
    """Check the PDF of Y(x1)=x1 is 1."""
    ptp = Ptp(Omega_X={'x1': (0, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=2)
    x1 = ptp.x_coords()

    density = ptp.fit(Y=x1, quadrature_degree=1000)
    f_tilde = density.pdf['f_tilde']
    assert assemble(abs(f_tilde-1)*dx) < 1e-8


def test_cdf_nonuniform_domain_length():
    """Check the CDF of Y(x1)=x1 is correctly calculated on shifted domain."""
    ptp = Ptp(Omega_X={'x1': (1, 2)}, Omega_Y={'Y': (1, 2)}, n_elements=10)
    x1 = ptp.x_coords()

    density = ptp.fit(Y=x1, quadrature_degree=500)
    assert assemble(((density.cdf-(density.y-1))**2)*dx) < 1e-8


def test_pdf_nonuniform_domain_length():
    """Check the PDF of Y(x1)=x1 is correctly calculated on shifted domain."""
    ptp = Ptp(Omega_X={'x1': (1, 2)}, Omega_Y={'Y': (1, 2)}, n_elements=2)
    x1 = ptp.x_coords()

    density = ptp.fit(Y=x1, quadrature_degree=1000)
    f_tilde = density.pdf["f_tilde"]
    assert assemble(abs(f_tilde-1)*dx) < 1e-8


def test_cdf_piecewise():
    """Test the CDF of a piecewise continuous function."""
    ptp = Ptp(Omega_X={'x1': (0, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=50)
    x1 = ptp.x_coords()

    expression = conditional(gt(x1, 1/2), x1, x1/2)
    density = ptp.fit(Y=expression, quadrature_degree=1000)
    
    y = ptp.y_coord()
    expression = conditional(le(y, 1/4), 2*y, 0) + conditional(And(gt(y, 1/4), le(y, 1/2)), 1/2, 0) + conditional(gt(y, 1/2), y, 0)
    F = Function(ptp.V_F)
    F.interpolate(expression)

    assert assemble(((F - density.cdf)**2)*dx) < 5e-6


def test_pdf_piecewise():
    """Test the PDF of a piecewise continuous function."""
    ptp = Ptp(Omega_X={'x1': (-1, 1)}, Omega_Y={'Y': (-0.25, 1)}, n_elements=5)
    x1 = ptp.x_coords()

    expression = conditional(gt(x1, 0), x1, 0)
    density = ptp.fit(Y=expression, quadrature_degree=1000)

    y = ptp.y_coord()
    expression = conditional(gt(y, 0), 1/2, 0)
    mesh = density.cdf.function_space().mesh()
    V_f = FunctionSpace(mesh=mesh, family="DG", degree=0)
    f = Function(V_f)
    f.interpolate(expression)

    assert assemble(abs(density.pdf['f_tilde'] - f)*dx ) < 1e-03


def test_cdf_quadratic():
    """Check the CDF of Y(x1)=x1^2 is correctly calculated on shifted domain."""
    ptp = Ptp(Omega_X={'x1': (0, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=100)
    x1 = ptp.x_coords()

    density = ptp.fit(Y=x1**2, quadrature_degree=1000)
    cdf = density.y**(1/2)
    assert assemble(((density.cdf-cdf)**2)*dx) < 1e-5


def test_pdf_quadratic():
    """Check the PDF of Y(x1)=x1^2 is correctly calculated on shifted domain."""
    ptp = Ptp(Omega_X={'x1': (0, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=100)
    x1 = ptp.x_coords()

    density = ptp.fit(Y=x1**2, quadrature_degree=1000)
    y = density.y
    int_num = density.distribution(y)

    # Analytic density
    #f = .5/y**(1/2) - Compare the mean due to singularity

    assert abs(int_num - 1./3.) < 1e-03


def test_cdf_cosine():
    """Check the CDF of Y(x1)=cos(x1)."""
    ptp = Ptp(Omega_X={'x1': (0, 2*np.pi)}, Omega_Y={'Y': (-1, 1)}, n_elements=100)
    x1 = ptp.x_coords()

    density = ptp.fit(Y=cos(x1), quadrature_degree=1000)
    cdf = 1 - acos(density.y)/np.pi
    assert assemble(((density.cdf-cdf)**2)*dx) < 1e-5


def test_pdf_cosine():
    """Check the PDF of Y(x1)=cos(x1)."""
    ptp = Ptp(Omega_X={'x1': (0, 2*np.pi)}, Omega_Y={'Y': (-1, 1)}, n_elements=100)
    x1 = ptp.x_coords()

    # Numerical
    density = ptp.fit(Y=cos(x1), quadrature_degree=1000)
    y = density.y
    
    # Analytical
    f = 1/(np.pi*(1-y**2)**.5)

    assert assemble((density.pdf['f_tilde'] - f)*dx ) < 1e-03


def test_qdf_uniform():
    """Check that Q( F(x) ) - x = 0 if Q = F^{-1} for Y(x1) = 0."""
    ptp = Ptp(Omega_X={'x1': (0, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=5)
    x1 = ptp.x_coords()

    # Act
    density = ptp.fit(Y=0*x1, quadrature_degree=1000)
    QF = density.compose(density.qdf, density.cdf, quadrature_degree=100)
    
    assert assemble((QF-density.y)*dx) < 1e-8


def test_qdf_straight_line():
    """Check that Q( F(x) ) - x = 0 if Q = F^{-1} for Y(x1) = x1."""
    # Arrange
    ptp = Ptp(Omega_X={'x1': (0, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=5)
    x1 = ptp.x_coords()

    # Act
    density = ptp.fit(Y=x1, quadrature_degree=1000)
    QF = density.compose(density.qdf, density.cdf, quadrature_degree=100)
    
    assert assemble((QF-density.y)*dx) < 1e-8


def test_qdf_piecewise():
    """Check that Q( F(x) ) - x = 0 for Y(x1) = { if x1 > 1/2: x1 else: x1/2."""
    ptp = Ptp(Omega_X={'x1': (0, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=50)
    x1 = ptp.x_coords()

    expression = conditional(gt(x1, 1/2), x1, x1/2)
    density = ptp.fit(Y=expression, quadrature_degree=1000)
    
    QF = density.compose(density.qdf, density.cdf, quadrature_degree=100)    
    assert assemble((QF-density.y)*dx) < 1e-8


def test_ape_rbc():
    """Validate the APE for RBC against its analytical value."""
    # Arrange
    ptp = Ptp(Omega_X={'x1': (-1, 1), 'x2': (0, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=5)
    x1, x2 = ptp.x_coords()

    # Construct the PDF,CDF,QDF for B
    p_B = ptp.fit(Y=1-x2, quadrature_degree=100)
    
    # Construct the PDF,CDF,QDF for Z
    p_Z = ptp.fit(Y=x2, quadrature_degree=100)

    # Act
    Z_ref = p_B.compose(p_Z.qdf, p_B.cdf, quadrature_degree=100)
    b = p_B.y

    bpe = p_B.distribution(b*Z_ref)
    tpe = (1/2)*assemble(x2*(1-x2)*dx)

    ape_numerical = bpe-tpe
    ape_exact = 1./6.

    assert abs(ape_numerical - ape_exact)/ape_exact < .005


def test_ape_layered():
    """Check APE for a layered stratification against its analytical value."""
    # Arrange
    ptp = Ptp(Omega_X={'x1': (-1, 1)}, Omega_Y={'Y': (-1, 1)}, n_elements=200)

    # Define the functions
    alpha = 1000
    x2 = ptp.x_coords()
    B = ( tanh(alpha*(x2 - 1/2)) + tanh(alpha*(x2 + 1/2)) )/2
    Z = x2

    # Construct the density objects
    density_B = ptp.fit(Y=B, quadrature_degree=2000)
    density_Z = ptp.fit(Y=Z, quadrature_degree=2000)

    # Define the map
    Q_B = density_B.qdf
    F_Z = density_Z.cdf
    beta_star = density_Z.compose(Q_B, F_Z, quadrature_degree=2000)

    # Act
    z = density_Z.y
    bpe_num = density_Z.distribution(beta_star*z)

    ape_numerical = bpe_num - 3/8
    ape_exact = 0.

    assert abs(ape_numerical - ape_exact) < .005
