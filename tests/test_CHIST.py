
# target = __import__("./PDF_Projector.py")
# ptp    = target.FEptp()

from PDF_Projector import *

# Unit Tests

def test_instantiation_function_spaces():

    FEptp(func_space_PDF= {"family":"CG","degree":3})

    assert True


def test_domain():

    # 1D
    FEptp(Omega_X = {'x1':(-1,1)}, Omega_Y = {'Y':(0,1)}, N_elements = 100)

    # 2D
    FEptp(Omega_X = {'x1':(-1,1),'x2':(-1,1)}, Omega_Y = {'Y':(0,1)}, N_elements = 10)

    assert True


# Integrated tests

## CDF tests
def test_CDF_constant():

    # Arrange
    ptp  = FEptp(Omega_X = {'x1':(0,1)}, Omega_Y = {'Y':(0,1)}, N_elements=10)
    
    # Act
    ptp.fit(function_Y = 0, quadrature_degree=10)
 
    assert assemble( ((ptp.F-1)**2)*dx ) < 1e-8;

def test_CDF_uniform_domain_length():

    # Arrange
    ptp  = FEptp(Omega_X = {'x1':(0,1)}, Omega_Y = {'Y':(0,1)}, N_elements=10)
    
    # Act
    x1,_ = ptp.coords
    ptp.fit(function_Y = x1, quadrature_degree=500)
 
    assert assemble( ((ptp.F-ptp.y)**2)*dx ) < 1e-8;

def test_CDF_nonuniform_domain_length():

    # Arrange
    ptp  = FEptp(Omega_X = {'x1':(1,2)}, Omega_Y = {'Y':(1,2)}, N_elements=10)
    
    # Act
    x1,_ = ptp.coords
    ptp.fit(function_Y = x1, quadrature_degree=1000)
 
    assert assemble( ((ptp.F-(ptp.y-1))**2)*dx ) < 1e-8;

# PDF tests
def test_PDF_uniform_domain_length():

    # Arrange
    ptp  = FEptp(Omega_X = {'x1':(0,1)}, Omega_Y = {'Y':(0,1)}, N_elements=10)
    
    # Act
    x1,_ = ptp.coords
    ptp.fit(function_Y = x1, quadrature_degree=1000)
 
    assert assemble( ((ptp.f-1.)**2)*dx ) < 1e-4;

def test_PDF_sine():

    raise NotImplementedError

## QDF tests
def test_QDF_uniform():

    ptp = FEptp(Omega_X = {'x1':(0,1)}, Omega_Y = {'Y':(0,1)}, N_elements=3)
    ptp.fit(function_Y  = 0, quadrature_degree=10)

    # Act
    quadrature_degree = 10
    y, = SpatialCoordinate(ptp.m_y)
    QF = ptp.compose(ptp.Q,ptp.F,quadrature_degree=quadrature_degree)
    
    # check that Q( F(x) ) - x = 0 if Q = F^{-1}
    assert assemble( (QF - y)*dx(quadrature_degree) ) < 1e-8;

def test_QDF_straight_line():

    # Arrange
    ptp = FEptp(Omega_X = {'x1':(0,1)}, Omega_Y = {'Y':(0,1)}, N_elements=3)
    x1,_= ptp.coords
    ptp.fit(function_Y = x1, quadrature_degree=10)

    # Act
    quadrature_degree = 10
    y, = SpatialCoordinate(ptp.m_y)
    QF = ptp.compose(ptp.Q,ptp.F,quadrature_degree=quadrature_degree)
    
    # check that Q( F(x) ) - x = 0 if Q = F^{-1}
    assert assemble( (QF - y)*dx(quadrature_degree) ) < 1e-8;

def test_QDF_piecewise():

    raise NotImplementedError

## APE tests
def test_APE_RBC():

    # Arrange

    # Construct the PDF,CDF,QDF for B
    ptp_B = FEptp(Omega_X = {'x1':(-1,1),'x2':(0,1)}, Omega_Y = {'Y':(0,1.)}, N_elements=10)
    x1,x2,_ = ptp_B.coords
    ptp_B.fit(function_Y = 1-x2, quadrature_degree=100) 

    # Construct the PDF,CDF,QDF for Z
    ptp_Z = FEptp(Omega_X = {'x1':(-1,1),'x2':(0,1)}, Omega_Y = {'Y':(0,1)}, N_elements=10)
    x1,x2,_ = ptp_Z.coords
    ptp_Z.fit(function_Y = x2, quadrature_degree=100)
    

    # Act
    
    quadrature_degree = 10
    Z_ref  = ptp_B.compose(ptp_Z.Q,ptp_B.F,quadrature_degree=quadrature_degree)
    b,     = SpatialCoordinate(ptp_B.m_y)
    BPE    = assemble( Z_ref*b*ptp_B.f*dx(degree=quadrature_degree) )
    TPE    = (1/2)*assemble( x2*(1-x2)*dx )

    APE_numerical = BPE-TPE
    APE_exact     = 1./6.

    assert abs(APE_numerical - APE_exact) < 1e-04

# def test_APE_Layers():
#     # Arrange
#     # Act
#     APE_numerical = BPE-TPE
#     APE_exact     =   
#     assert abs(APE_numerical - APE_exact) < 1e-08