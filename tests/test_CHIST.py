from firedrake import *
from chist import Ptp


def test_domain():

    # 1D
    Ptp(Omega_X={'x1': (-1, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=100)

    # 2D
    Ptp(Omega_X={'x1': (-1, 1), 'x2': (-1, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=10)

    assert True

# CDF tests
def test_CDF_constant():

    # Arrange
    ptp = Ptp(Omega_X={'x1': (0, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=10)
    x1 = ptp.x_coords()
    
    # Act
    density = ptp.fit(Y=0*x1)
    assert assemble(((density.cdf-1)**2)*dx) < 1e-8

def test_CDF_uniform_domain_length():

    # Arrange
    ptp = Ptp(Omega_X={'x1': (0, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=10)
    x1 = ptp.x_coords()

    # Act
    density = ptp.fit(Y=x1, quadrature_degree=500)
    assert assemble(((density.cdf-density.y)**2)*dx) < 1e-8

def test_CDF_nonuniform_domain_length():

    # Arrange
    ptp = Ptp(Omega_X={'x1': (1, 2)}, Omega_Y={'Y': (1, 2)}, n_elements=10)
    x1 = ptp.x_coords()

    # Act
    density = ptp.fit(Y=x1, quadrature_degree=500)
    assert assemble(((density.cdf-(density.y-1))**2)*dx) < 1e-8

# PDF tests
def test_PDF_uniform_domain_length():

    # Arrange
    ptp = Ptp(Omega_X={'x1': (0, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=5)
    x1 = ptp.x_coords()
    
    # Act
    density = ptp.fit(Y=x1, quadrature_degree=1000)
    assert assemble(((density.pdf-1.)**2)*dx) < 1e-8

# QDF tests
def test_QDF_uniform():
    """Check that Q( F(x) ) - x = 0 if Q = F^{-1}."""
    ptp = Ptp(Omega_X={'x1': (0, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=5)
    x1 = ptp.x_coords()

    # Act
    density = ptp.fit(Y=0*x1, quadrature_degree=1000)
    QF = density.compose(density.qdf, density.cdf, quadrature_degree=10)
    
    assert assemble((QF-density.y)*dx(10)) < 1e-8

def test_QDF_straight_line():
    """Check that Q( F(x) ) - x = 0 if Q = F^{-1}."""

    # Arrange
    ptp = Ptp(Omega_X={'x1': (0, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=5)
    x1 = ptp.x_coords()

    # Act
    density = ptp.fit(Y=x1, quadrature_degree=1000)
    QF = density.compose(density.qdf, density.cdf, quadrature_degree=10)
    
    assert assemble((QF-density.y)*dx(10)) < 1e-8


## APE tests
def test_APE_RBC():
    """Validate the APE for RBC against its analytical value."""
    # Arrange
    ptp = Ptp(Omega_X={'x1': (-1, 1), 'x2': (0, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=10)
    x1,x2 = ptp.x_coords()
    
    # Construct the PDF,CDF,QDF for B
    p_B = ptp.fit(Y=1-x2, quadrature_degree=100)

    # Construct the PDF,CDF,QDF for Z
    p_Z = ptp.fit(Y=x2, quadrature_degree=100)

    # Act
    z_ref = p_B.compose(p_Z.qdf, p_B.cdf, quadrature_degree=10)
    b = p_B.y
    bpe = assemble(z_ref*b*p_B.pdf*dx(degree=10))
    tpe = (1/2)*assemble( x2*(1-x2)*dx )

    ape_numerical = bpe-tpe
    ape_exact     = 1./6.

    assert abs(ape_numerical - ape_exact) < 1e-04

'''
def test_QDF_piecewise():

    raise NotImplementedError

# def test_APE_Layers():
#     # Arrange
#     # Act
#     APE_numerical = BPE-TPE
#     APE_exact     =   
#     assert abs(APE_numerical - APE_exact) < 1e-08


def test_PDF_sine():

    raise NotImplementedError
'''