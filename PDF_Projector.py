"""
class which given a user provided "function" over a physical "domain"

    Y(x) where x in Ω

returns the "CDF" & "PDF"

    F_Y(y), f_Y(y)

over its corresponding probability space  ( Ω_Y, Y, I) using a finite 
element discretisation consisting of n_bins
"""

# Currently the range of the function Y(x) only goes from 0,1 how to change this

# How to recover F from F_hat if the mesh is triangles rather than an interval?

# Would it be more efficient to import only certain parts of the library??
import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
import pyvista as pv


class FEptp(object):

    def __init__(self, func_space_CDF = {"family":"DG","degree":1},func_space_PDF= {"family":"CG","degree":1}):

        # Physical space
        self.Y  = None
        self.Xdim = None
        
        # Mesh
        self.m_y  = None
        self.m_yx = None

        # Finite Elements
        self.V_FE = FiniteElement(family=func_space_CDF['family'],cell="interval",degree=func_space_CDF['degree'])
        self.V_fE = FiniteElement(family=func_space_PDF['family'],cell="interval",degree=func_space_PDF['degree'])
        
        # Function spaces
        self.V_f     = None
        self.V_F     = None
        self.V_F_hat = None

        # CDF,PDF,y
        self.F = None
        self.f = None
        self.y = None

        return None;

    def indicator(self):

        """
        Define the indicator function I(y,x=(x1,x2)) which acts on the random function Y(x1,x2)
        """
        if self.Xdim == 1:
            x1,   y = SpatialCoordinate(self.m_yx)
        elif self.Xdim == 2:
            x1,x2,y = SpatialCoordinate(self.m_yx)

        I  = conditional( self.Y < y, 1.,0.)

        return I;

    def domain(self, Omega = {'x1':(-1,1),'x2':(-1,1)}, N_elements = 10):
        
        """
        Construct the extruded mesh 
            Ω x Ω_Y
        the physical space times the event space
        """
        self.Xdim = len(Omega)

        # x-direction
        if   self.Xdim == 1:
            print('1D mesh')
            cell_type = "interval";  
            mesh_x    = IntervalMesh(ncells=1,length_or_left=Omega['x1'][0],right=Omega['x1'][1])
        elif self.Xdim == 2:
            print('2D mesh')
            cell_type = "triangle";
            mesh_x    = RectangleMesh(nx=1,ny=1,Lx=Omega['x1'][1],Ly=Omega['x2'][1],originX=Omega['x1'][0],originY=Omega['x2'][0])
        else:
            raise ValueError('The domain Ω must be 1D or 2D \n')

        R         = FiniteElement(family="DG",cell=cell_type,degree=0)
        T_element = TensorProductElement(R,self.V_FE)

        # Add y-direction
        self.m_y  = IntervalMesh(        ncells=N_elements,length_or_left=0,right=1) 
        self.m_yx = ExtrudedMesh(mesh_x, layers=N_elements,layer_height=1./N_elements,extrusion_type='uniform')
        
        File("Exruded_Mesh.pvd").write(self.m_yx)

        # Function-spaces
        self.V_f     = FunctionSpace(mesh=self.m_y ,family=self.V_fE)
        self.V_F     = FunctionSpace(mesh=self.m_y ,family=self.V_FE)
        self.V_F_hat = FunctionSpace(mesh=self.m_yx,family=T_element) # extension of V_F into x     
        
        return SpatialCoordinate(self.m_yx)

    def CDF(self,quadrature_degree):

        """
        Construct the CDF F_Y(y) of the random function Y(x)
        by projecting from physical space into the probability 
        space specified.
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
        ys  = interpolate(y,self.V_F)
        indx= np.argsort(ys.dat.data)

        # Pass F_hat into F
        if len(F_hat.dat.data) == len(indx):
            self.F.dat.data[indx] = F_hat.dat.data[:]
        else:
            self.F.dat.data[indx] = 0.5*(F_hat.dat.data[:len(indx)] + F_hat.dat.data[len(indx):])

        print('CDF int F ds = ',assemble(self.F*ds),'\n')

        return None;

    def PDF(self):

        """
        Construct the PDF f_Y(y) = ∂y F_Y(y) of the random function Y(x)
        by constructing a projecting of the above relation and putting the 
        the derivative onto the test function.
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

        # Check it integrates to 1
        print('int f dx = ',assemble(self.f*dx),'\n')

        return None;    

    def fit(self,function_Y,quadrature_degree=500):

        """
        
        """        
        self.Y = function_Y # Assign input
        self.CDF(quadrature_degree) # Solve for the CDF
        self.PDF() # Solve for the PDF

        return None;

    def evaluate(self,y):

        raise NotImplementedError
    
    def plot(self):
        
        """
        
        """

        try:
            Line2D_F = plot(self.F,num_sample_points=50)
            plt.title(r'CDF',fontsize=20)
            plt.ylabel(r'$F_Y$',fontsize=20)
            plt.xlabel(r'$y$',fontsize=20)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            warning("Cannot plot figure. Error msg: '%s'" % e)

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

    # Convinience functions
    def vis_indicator(self):

        # Visualisation of the indicator function
        N_vis = 50
        
        # x-direction
        if self.Xdim == 1:
            cell_type = "interval";  
            mI_x = IntervalMesh(N_vis,0,1); 
        elif self.Xdim == 2:
            cell_type = "triangle";
            mI_x  = RectangleMesh(N_vis,N_vis,Lx=2*np.pi,Ly=np.pi, originX=0.0, originY=0.0);
        # y-direction
        meshI = ExtrudedMesh(mI_x, layers=N_vis) 

        # Set the function-space V_I
        W   = FiniteElement(family="DG",cell=cell_type ,degree=0)
        V_F = FiniteElement(family="DG",cell="interval",degree=0)

        elt_WVF = TensorProductElement(W,V_F)
        V_I     = FunctionSpace(meshI,elt_WVF)

        I = Function(V_I)
        I.interpolate( self.indicator() )
        File("Indicator_Function.pvd").write(I)

        reader = pv.get_reader(filename="Indicator_Function.pvd")
        fdrake_mesh = reader.read()[0]
        fdrake_mesh.plot(cmap='coolwarm') #cpos='xy'

        return None;

    def __str__(self):
        s= ( 'Continuation succeed \n');
        return s
        
if __name__ == "__main__":

    # %%    
    print("Initialising library \n")

    # %%
    ptp   = FEptp()

    x1,x2,y = ptp.domain(Omega = {'x1':(0,2*np.pi),'x2':(0,2*np.pi)}, N_elements=10)
    ptp.fit(function_Y = sin(x1)**2 + sin(x2)**2, quadrature_degree=150)

    # x1,y = ptp.domain(Omega = {'x1':(0,1)}, N_elements=10)
    # ptp.fit(function_Y = sin(x1)**2, quadrature_degree=200)

    ptp.plot()
    print("Testing")