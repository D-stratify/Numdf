"""
FEptp class - Finite Element physical to probability 

Given a user provided "function" over a physical "domain"

    Y(X) where X in Ω_X

this class uses the fit method to return the "CDF", "QDF" & "PDF"

    F_Y(y), Q_Y(p), f_Y(y)

over their corresponding probability space Ω_Y. The method uses 
a finite element discretisation consisting of n elements (bins).
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
from firedrake import *
import numpy as np
from firedrake.__future__ import interpolate

class FEptp(object):

    def __init__(self, func_space_CDF = {"family":"DG","degree":1},func_space_PDF= {"family":"CG","degree":1}):

        # Physical space
        self.Y  = None
        self.Xdim = None
        
        # Mesh
        self.m_y  = None
        self.m_yx = None
        self.m_p  = None

        # Finite Elements
        variant   = "equispaced" #"spectral"
        self.V_FE = FiniteElement(family=func_space_CDF['family'],cell="interval",degree=func_space_CDF['degree'],variant=variant)
        self.V_fE = FiniteElement(family=func_space_PDF['family'],cell="interval",degree=func_space_PDF['degree'],variant=variant)
        
        # Function spaces
        self.V_f     = None
        self.V_F     = None
        self.V_F_hat = None

        # CDF,QDF,PDF,y
        self.F = None
        self.Q = None
        self.f = None
        self.y = None

        return None;

    def indicator(self):

        """
        Defines the indicator function I(y,x=(x1,x2)) which acts on the random function Y(x1,x2)
        """

        if self.Xdim == 1:
            x1,   y = SpatialCoordinate(self.m_yx)
        elif self.Xdim == 2:
            x1,x2,y = SpatialCoordinate(self.m_yx)

        I  = conditional( self.Y < y, 1.,0.)

        return I;

    def domain(self, Omega_X = {'x1':(-1,1),'x2':(-1,1)}, Omega_Y = {'Y':(0,1)}, N_elements = 10):
        
        """
        Constructs the extruded mesh Ω_X x Ω_Y given by the physical space Ω_X times the event space Ω_Y

        Inputs: 

        Omega_X 'dict' - Physical domain 
        Omega_Y 'dict' - Probability space 
        N_elements int - Number of finite elements 
            
        """

        self.Xdim = len(Omega_X)

        # x-direction
        if   self.Xdim == 1:
            print('1D mesh')
            cell_type = "interval";  
            mesh_x    = IntervalMesh(ncells=1,length_or_left=Omega_X['x1'][0],right=Omega_X['x1'][1])
        elif self.Xdim == 2:
            print('2D mesh')
            cell_type = "triangle";
            #cell_type = "quadrilateral"
            mesh_x    = RectangleMesh(nx=1,ny=1,Lx=Omega_X['x1'][1],Ly=Omega_X['x2'][1],originX=Omega_X['x1'][0],originY=Omega_X['x2'][0])
        else:
            raise ValueError('The domain Ω must be 1D or 2D \n')

        R         = FiniteElement(family="DG",cell=cell_type,degree=0)
        T_element = TensorProductElement(R,self.V_FE)

        # Add y-direction
        self.m_y  = IntervalMesh(        ncells=N_elements,length_or_left=Omega_Y['Y'][0],right=Omega_Y['Y'][1]) 
        self.m_yx = ExtrudedMesh(mesh_x, layers=N_elements,layer_height=1./N_elements,extrusion_type='uniform')
        
        File("Exruded_Mesh.pvd").write(self.m_yx)

        # Function-spaces
        self.V_f     = FunctionSpace(mesh=self.m_y ,family=self.V_fE)
        self.V_F     = FunctionSpace(mesh=self.m_y ,family=self.V_FE)
        self.V_F_hat = FunctionSpace(mesh=self.m_yx,family=T_element) # extension of V_F into x   

        # Mapping 
        self.Omega_Y_to_01 = lambda Y:  Y/(Omega_Y['Y'][1]-Omega_Y['Y'][0]) - Omega_Y['Y'][0]/(Omega_Y['Y'][1]-Omega_Y['Y'][0])

        return SpatialCoordinate(self.m_yx)

    def CDF(self,quadrature_degree):

        """
        Construct the CDF F_Y(y) of the random function Y(x) by projecting from physical space into the probability space specified.

        Inputs:

        quadrature_degree int - order of the numerical quadrature scheme to use

        """
        
        # Define trial & test functions on V_F_hat
        u = TrialFunction(self.V_F_hat)
        v = TestFunction( self.V_F_hat)

        # Construct the linear & bilinear forms
        a = inner(u,v) * dx
        L = inner(self.indicator(),v) * dx(degree=quadrature_degree)

        # Solve for F_hat
        F_hat = Function(self.V_F_hat)
        solve(a == L,F_hat)  

        # Recover F_Y(y) in V_F
        self.F = Function(self.V_F)

        # Sort a linear function in ascending order 
        # this creates a DOF map which matches 
        # the extended mesh which are in ascending order
        y,  = SpatialCoordinate(self.m_y)
        ys  = assemble(interpolate(y,self.V_F))
        indx= np.argsort(ys.dat.data)

        # Pass F_hat into F
        if len(F_hat.dat.data) == len(indx):
            self.F.dat.data[indx] = F_hat.dat.data[:]
        else:
            #print(len(indx))
            #print(len(F_hat.dat.data))
            self.F.dat.data[indx] = 0.5*(F_hat.dat.data[:len(indx)] + F_hat.dat.data[len(indx):])

        # Check CDF properties
        Surf_int = assemble(self.F*ds)
        if abs(Surf_int - 1) > 1e-02:
            print("Calculated F(+∞) - F(-∞) should equal 1, got %e. Check the domain of Ω_Y and the quadrature_degree specified."%Surf_int)
            #raise ValueError("Calculated F(+∞) - F(-∞) should equal 1, got %e. Check the domain of Ω_Y and the quadrature_degree specified."%Surf_int)

        return None;

    def QDF(self):
        """
        Construct the QDF (inverse CDF) Q_Y(y) of the random function Y(x) by inverting F_Y(y)
        """

        # From the CDF F obtain the F_i values
        F_i  = self.F.dat.data[:] 
    
        # Grab the Z_i values
        m_z = self.V_F.mesh()
        W   = VectorFunctionSpace(m_z, self.V_F.ufl_element())
        Z   = assemble(interpolate(m_z.coordinates, W)).dat.data
        
        # Append bcs to F as by definition a CDF is 0,1 at -/+ infty 
        p         = np.hstack(( [0]     ,F_i,[1]       ))  
        
        # Append bcs to y 
        mesh = self.F.function_space().mesh()
        y_i  = mesh.coordinates.dat.data[:] 
        z    = np.zeros(2*len(Z))
        z[0:-1:2] = Z
        z[1:  :2] = Z
        z         = np.hstack(( [y_i[0]],z,[y_i[-1]] )) 

        # Make a 1D mesh whose vertices are given by the p values
        layers   = len(p[1:] - p[:-1]);
        self.m_p = UnitIntervalMesh(ncells=layers);
        self.m_p.coordinates.dat.data[:] = p[:]

        # Create a function Q(p) on this mesh
        self.V_Q  = FunctionSpace(mesh=self.m_p,family=self.V_FE)
        self.Q    = Function(self.V_Q)

        # Assign Q(p_i) = Q_i
        self.Q.dat.data[:] = z[:]

        return None;

    def PDF(self):

        """
        Construct the PDF f_Y(y) of the random function Y(x) by projecting f_Y(y) = ∂y F_Y(y)
        """

        # Define trial & test functions on V_f
        u = TrialFunction(self.V_f)
        v = TestFunction(self.V_f)

        # Construct the linear & bilinear forms
        a =  inner(u,v) * dx
        L = -inner(self.F,v.dx(0)) * dx  +  self.F*v*ds(2) - self.F*v*ds(1)

        # Solve for f
        self.f = Function(self.V_f)
        solve(a == L, self.f)

        # Check PDF properties
        PDF_int = assemble(self.f*dx)
        if abs(PDF_int - 1) > 1e-02:
            print("Calculated ∫ f(y) dy should equal 1, but got %e. Check the quadrature_degree used. "%PDF_int)
            #raise ValueError("Calculated ∫ f(y) dy should equal 1, but got %e. Check the quadrature_degree used. "%PDF_int)

        return None;    

    def fit(self,function_Y,quadrature_degree=500):

        """
        Constructs the CDF F_Y(y) and the PDF f_Y(y) of the function Y(X)

        Inputs:

            function_Y 'ufl expression' - the random function Y(X)
            quadrature_degree int - order of the numerical quadrature scheme to use
        """     

        # Assign input & map to Y \in Ω_Y |-> [0,1]   
        self.Y = self.Omega_Y_to_01(function_Y) 
        
        # Solve for the CDF, PDF and map back y \in [0,1] |-> Ω_Y 
        self.CDF(quadrature_degree)
        self.QDF()
        self.PDF()

        return None;

    def plot(self,function = 'CDF'):
        
        """
        Visualise the CDF, QDF and PDF using the inbuilt plotting routines
        """

        import matplotlib.pyplot as plt
        from firedrake.pyplot import plot

        if function == 'CDF':
            try:
                Line2D_F = plot(self.F,num_sample_points=50)
                plt.title(r'CDF',fontsize=20)
                plt.ylabel(r'$F_Y$',fontsize=20)
                plt.xlabel(r'$y$',fontsize=20)
                plt.tight_layout()
                plt.grid()
                plt.show()
            except Exception as e:
                warning("Cannot plot figure. Error msg: '%s'" % e)
        
        elif function == 'QDF':

            try:
                Line2D_F = plot(self.Q,num_sample_points=50)
                plt.title(r'QDF',fontsize=20)
                plt.ylabel(r'$Q_Y$',fontsize=20)
                plt.xlabel(r'$p$',fontsize=20)
                plt.tight_layout()
                plt.grid()
                plt.show()
            except Exception as e:
                warning("Cannot plot figure. Error msg: '%s'" % e)
        
        elif function == 'PDF':

            try:
                Line2D_f = plot(self.f,num_sample_points=50)
                plt.title(r'PDF',fontsize=20)
                plt.ylabel(r'$f_Y$',fontsize=20)
                plt.xlabel(r'$y$',fontsize=20)
                plt.tight_layout()
                plt.show()
            except Exception as e:
                warning("Cannot plot figure. Error msg: '%s'" % e)

        return None

    def evaluate(self,y):

        """
        Returns the CDF and PDF evaluated an the point(s) y

        Inputs:
            y 'array or list' - [0, 0.1, ... ,1] the locations to evaluate 
        
        Returns:
            F_Y- 'array' CDF evaluated at y
            f_Y- 'array' PDF evaluated at y
            y  - 'array' locations y

        """

        F_Y = np.asarray(self.F.at(y))

        f_Y = np.asarray(self.f.at(y))

        y_i = np.asarray(y)

        return F_Y,f_Y,y_i;
    
    def __str__(self):

        """
        Print details of the FEptp object
        """
        
        s= ( 'Approximation spaces: \n'
            + 'CDF F_Y(y) \n'
            + 'PDF f_Y(y) \n'
            + 'domain Ω \n'
            + 'N elements \n');
        
        return s
        
if __name__ == "__main__":

    #%%
    print("Initialise")
    
    # %%
    # Specifiy the function spaces for the CDF & PDF
    ptp = FEptp(func_space_CDF = {"family":"DG","degree":1},func_space_PDF= {"family":"CG","degree":1})

    # (a) Specify the domain size(s) and number of finite elements/bins 
    # (b) Projection Y(X) into probability space
    # (c) Plot out the functions

    # 1D example
    x1,y = ptp.domain(Omega_X = {'x1':(0,1)}, Omega_Y = {'Y':(0,1)}, N_elements=50)
    ptp.fit(function_Y = x1**(3/2), quadrature_degree=1000)
    ptp.plot()

    #2D example
    x1,x2,y = ptp.domain(Omega_X = {'x1':(0,1),'x2':(0,1)}, Omega_Y = {'Y':(0,2)}, N_elements=50)
    ptp.fit(function_Y = x1 + x2, quadrature_degree=500)
    ptp.plot()

    # Evaluate the CDF & PDF at points
    F_Y,f_Y,y_i  = ptp.evaluate(y = [0., 0.1, 0.2])