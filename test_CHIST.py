
# target = __import__("./PDF_Projector.py")
# ptp    = target.FEptp()

from PDF_Projector import *

# Unit Tests

def test_instantiation_function_spaces():

    ptp  = FEptp(func_space_CDF = {"family":"DG","degree":2},func_space_PDF= {"family":"CG","degree":3})

    assert True


def test_domain():

    FEptp().domain( Omega_X = {'x1':(-1,1),'x2':(-1,1)}, Omega_Y = {'Y':(0,1)}, N_elements = 10)

    FEptp().domain( Omega_X = {'x1':(-1,1)}, Omega_Y = {'Y':(0,1)}, N_elements = 100)

    assert True


# Integrated tests

def test_uniform():

    # Arrange
    ptp  = FEptp()
    x1,y = ptp.domain(Omega_X = {'x1':(0,1)}, Omega_Y = {'Y':(0,1)}, N_elements=10)
    
    # Act
    ptp.fit(function_Y = x1, quadrature_degree=1000)
 
    assert assemble( ((ptp.F-ptp.y)**2)*dx ) < 1e-8;

def test_uniform_domain_length():

    # Arrange
    ptp  = FEptp()
    x1,y = ptp.domain(Omega_X = {'x1':(1,2)}, Omega_Y = {'Y':(1,2)}, N_elements=10)
    
    # Act
    ptp.fit(function_Y = x1, quadrature_degree=1000)
 
    assert assemble( ((ptp.F-(ptp.y-1))**2)*dx ) < 1e-8;

def test_constant():

    # Arrange
    ptp  = FEptp()
    x1,y = ptp.domain(Omega_X = {'x1':(0,1)}, Omega_Y = {'Y':(0,1)}, N_elements=10)
    
    # Act
    ptp.fit(function_Y = 0, quadrature_degree=10)
 
    assert assemble( ((ptp.F-1)**2)*dx ) < 1e-8;


# Convergence tests pass - use sine function
