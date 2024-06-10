from firedrake import *
from chist import Ptp


def test_domain():
    """Ensure we can import and initialise Ptp."""
    # 1D
    Ptp(Omega_X={'x1': (-1, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=2)

    # 2D
    Ptp(Omega_X={'x1': (-1, 1), 'x2': (-1, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=2)

    assert True


def test_cdf_constant():
    """Check the CDF of Y(X)=0 is correctly calculated."""
    # Arrange
    ptp = Ptp(Omega_X={'x1': (0, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=2)
    x1 = ptp.x_coords()
    
    # Act
    density = ptp.fit(Y=0*x1)
    assert assemble(((density.cdf-1)**2)*dx) < 1e-8


def test_cdf_uniform_domain_length():
    """Check the CDF of Y(x1)=x1 is y"""
    # Arrange
    ptp = Ptp(Omega_X={'x1': (0, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=10)
    x1 = ptp.x_coords()

    # Act
    density = ptp.fit(Y=x1, quadrature_degree=500)
    assert assemble(((density.cdf-density.y)**2)*dx) < 1e-8


def test_cdf_nonuniform_domain_length():
    """Check the CDF of Y(x1)=x1 is correctly calculated on shifted domain."""
    # Arrange
    ptp = Ptp(Omega_X={'x1': (1, 2)}, Omega_Y={'Y': (1, 2)}, n_elements=10)
    x1 = ptp.x_coords()

    # Act
    density = ptp.fit(Y=x1, quadrature_degree=500)
    assert assemble(((density.cdf-(density.y-1))**2)*dx) < 1e-8


def test_cdf_piecewise():
    """Test the CDF of a piecewise continuous function."""
    ptp = Ptp(Omega_X={'x1': (0, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=100)
    x1 = ptp.x_coords()

    expression = conditional( gt(x1, 1/2), x1, x1/2)
    density = ptp.fit(Y=expression, quadrature_degree=1000)
    
    y = ptp.y_coord()
    expression = conditional(le(y, 1/4), 2*y, 0) + conditional(And(gt(y, 1/4), le(y, 1/2)), 1/2, 0) + conditional(gt(y, 1/2), y, 0)
    F = Function(ptp.V_F)
    F.interpolate(expression)

    assert assemble(((F - density.cdf)**2)*dx) < 1e-6


def test_pdf_uniform_domain_length():
    """Check the pdf of Y(x1)=x1 is 1."""
    # Arrange
    ptp = Ptp(Omega_X={'x1': (0, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=5)
    x1 = ptp.x_coords()
    
    # Act
    density = ptp.fit(Y=x1, quadrature_degree=1000)
    assert assemble(((density.pdf-1.)**2)*dx) < 1e-8


def test_qdf_uniform():
    """Check that Q( F(x) ) - x = 0 if Q = F^{-1} for Y(x1) = 0."""
    ptp = Ptp(Omega_X={'x1': (0, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=5)
    x1 = ptp.x_coords()

    # Act
    density = ptp.fit(Y=0*x1, quadrature_degree=1000)
    QF = density.compose(density.qdf, density.cdf, quadrature_degree=10)
    
    assert assemble((QF-density.y)*dx(10)) < 1e-8


def test_qdf_straight_line():
    """Check that Q( F(x) ) - x = 0 if Q = F^{-1} for Y(x1) = x1."""
    # Arrange
    ptp = Ptp(Omega_X={'x1': (0, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=5)
    x1 = ptp.x_coords()

    # Act
    density = ptp.fit(Y=x1, quadrature_degree=1000)
    QF = density.compose(density.qdf, density.cdf, quadrature_degree=10)
    
    assert assemble((QF-density.y)*dx(10)) < 1e-8


def test_qdf_piecewise():
    """Check that Q( F(x) ) - x = 0 for Y(x1) = { if x1 > 1/2: x1 else: x1/2."""
    ptp = Ptp(Omega_X={'x1': (0, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=200)
    x1 = ptp.x_coords()

    expression = conditional( gt(x1, 1/2), x1, x1/2)
    density = ptp.fit(Y=expression, quadrature_degree=1000)
    V_Q = density.qdf.function_space()
    m_p = V_Q.mesh()
    p = SpatialCoordinate(m_p)[0]
    expression = conditional(le(p, 1/2), p/2, p)
    Q = Function(V_Q)
    Q.interpolate(expression)

    assert assemble(((Q - density.qdf)**2)*dx ) < 1e-03


def test_ape_rbc():
    """Validate the APE for RBC against its analytical value."""
    # Arrange
    ptp = Ptp(Omega_X={'x1': (-1, 1), 'x2': (0, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=10)
    x1,x2 = ptp.x_coords()
    
    # Construct the PDF,CDF,QDF for B
    p_B = ptp.fit(Y=1-x2, quadrature_degree=200)
    
    # Construct the PDF,CDF,QDF for Z
    p_Z = ptp.fit(Y=x2, quadrature_degree=200)

    # Act
    z_ref = p_B.compose(p_Z.qdf, p_B.cdf, quadrature_degree=100)
    b = p_B.y
    bpe = assemble(z_ref*b*p_B.pdf*dx(degree=100))
    tpe = (1/2)*assemble( x2*(1-x2)*dx )

    ape_numerical = bpe-tpe
    ape_exact = 1./6.

    assert abs(ape_numerical - ape_exact) < 1e-03

'''
def test_PDF_sine():

    raise NotImplementedError

def test_APE_Layers():
    # Arrange
    # Act
    APE_numerical = BPE-TPE
    APE_exact     =   
    assert abs(APE_numerical - APE_exact) < 1e-08
'''