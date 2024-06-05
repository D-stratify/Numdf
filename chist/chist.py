"""
FEptp class - Finite Element physical to probability 

Given a user provided "function" over a physical "domain"

    Y(X) where X in Ω_X

this class uses the fit method to return the "CDF", "QDF" & "PDF"

    F_Y(y), Q_Y(p), f_Y(y)

over their corresponding probability space Ω_Y. The method uses 
a finite element discretisation consisting of n elements (bins).
"""

import os, copy
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from firedrake import *
from firedrake.__future__ import interpolate


class FEptp(object):

    def __init__(self, Omega_X = {'x1':(-1,1),'x2':(-1,1)}, Omega_Y = {'Y':(0,1)}, N_elements = 10,func_space_PDF= {"family":"CG","degree":1}):

        # Physical space
        self.Ω_X = Omega_X
        self.Ω_Y = Omega_Y
        self.N_e = N_elements
 
        # Mesh & Coordinates
        if   len(self.Ω_X) == 1:
            cell_type = "interval"
        elif len(self.Ω_X) == 2:
            cell_type = "quadrilateral"

        self.coords = self.domain()
        self.y,     = SpatialCoordinate(self.m_y)

        # Finite Elements
        variant   = "equispaced" #"spectral"
        self.R_FE = FiniteElement(family="DG",cell=cell_type ,degree=0,variant=variant)
        self.V_FE = FiniteElement(family="DG",cell="interval",degree=1,variant=variant)
        self.V_QE = FiniteElement(family="CG",cell="interval",degree=1,variant=variant)
        self.V_fE = FiniteElement(family=func_space_PDF['family'],cell="interval",degree=func_space_PDF['degree'],variant=variant)
        
        # Function-space  
        self.V_F     = FunctionSpace(mesh=self.m_y ,family=self.V_FE)
        T_element    = TensorProductElement(self.R_FE,self.V_FE)
        self.V_F_hat = FunctionSpace(mesh=self.m_yx,family=T_element) # extension of V_F into x
        self.V_f     = FunctionSpace(mesh=self.m_y ,family=self.V_fE)

        return None;

    def domain(self):
        
        """
        Constructs the extruded mesh Ω_X x Ω_Y given by the physical space Ω_X times the event space Ω_Y        
        """

        # x-direction
        if   len(self.Ω_X) == 1:
            self.mesh_x = IntervalMesh(ncells=1,length_or_left=self.Ω_X['x1'][0],right=self.Ω_X['x1'][1])
        elif len(self.Ω_X) == 2:
            self.mesh_x = RectangleMesh(nx=1,ny=1,Lx=self.Ω_X['x1'][1],Ly=self.Ω_X['x2'][1],originX=self.Ω_X['x1'][0],originY=self.Ω_X['x2'][0],quadrilateral=True)
        else:
            raise ValueError('The domain Ω must be 1D or 2D \n')

        # Add y-direction
        self.m_y  = IntervalMesh(        ncells=self.N_e,length_or_left=self.Ω_Y['Y'][0],right=self.Ω_Y['Y'][1]) 
        self.m_yx = ExtrudedMesh(self.mesh_x, layers=self.N_e,layer_height=1./self.N_e,extrusion_type='uniform')
        
        return SpatialCoordinate(self.m_yx)

    def map(self,Y):

        """
        Mapping of  Y(x) \in Ω_x    to      Y(X) \in [0,1]   
        """

        return Y/(self.Ω_Y['Y'][1]-self.Ω_Y['Y'][0]) - self.Ω_Y['Y'][0]/(self.Ω_Y['Y'][1]-self.Ω_Y['Y'][0])

    def fit(self,function_Y,quadrature_degree=500):

        """
        Constructs the CDF F_Y(y) and the PDF f_Y(y) of the function Y(X)

        Inputs:

            function_Y 'ufl expression' - the random function Y(X)
            quadrature_degree int - order of the numerical quadrature scheme to use
        """     

        # Map the function to [0,1]
        Y = self.map(function_Y)

        # Density obejct
        P = DENSITY(self,Y,quadrature_degree)

        # Generate CDF,QDF,PDF
        P.fit()

        return P
    
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


class DENSITY(object):

    def __init__(self, ptp,Y,quadrature_degree):

        self.ptp = ptp
        self.Y   = Y
        self.quadrature_degree = quadrature_degree

        # CDF,QDF,PDF
        self.F = None
        self.Q = None
        self.f = None

        return None

    def indicator(self):

        """
        Defines the indicator function I(y,x=(x1,x2)) which acts on the random function Y(x1,x2)
        """

        if len(self.ptp.Ω_X) == 1:
            x1,   y = self.ptp.coords
        elif len(self.Ω_X) == 2:
            x1,x2,y = self.ptp.coords

        return conditional( self.Y < y, 1.,0.)

    def slope_limiter(self):

        def jump_condition(a_n_minus,a_n_plus,a_0_minus): 
        
            if a_n_plus < a_n_minus: 
                return a_n_plus-a_n_minus 
            else:
                return min(a_n_plus,a_0_minus) - a_n_minus

        def jumps(F,F_0):

            celldata_0 = F_0.dat.data[:].reshape((-1,2))

            celldata_n = F.dat.data[:].reshape((-1,2))
            Ne         = celldata_n.shape[0]
            jumps      = np.zeros(Ne)

            # Go through the cells from left to right
            for e in range(Ne):

                # (1) cell data
                # e - 1
                if e == 0:
                    cell_n_em1= np.zeros(2)
                    cell_0_em1= np.zeros(2)
                else:
                    cell_n_em1= celldata_n[e-1,:]
                    cell_0_em1= celldata_0[e-1,:]
                # e
                cell_n_e = celldata_n[e,:] 
                cell_0_e = celldata_0[e,:]

                # e + 1
                if e == Ne-1:
                    cell_n_ep1= np.ones(2)
                else:
                    cell_n_ep1= celldata_n[e+1,:] 
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # (2) jumps 
                left     = jump_condition(cell_n_em1[1],cell_n_e[0],    cell_0_em1[1])
                right    = jump_condition(cell_n_e[1],cell_n_ep1[0],    cell_0_e[1])
                jumps[e] = min(left,right)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            return jumps

        F_0 = copy.deepcopy(self.F)
        Ne  = F_0.dat.data[:].reshape((-1,2)).shape[0]

        # A) Relaxation loop
        error = 1.
        iter  = 0.
        slope =-1. 
        Jo    = np.zeros(Ne)
        α = 0.1
        while (error > 0.2) or (slope < 0):

            # (1) Update dats
            Jn = jumps(self.F,F_0)
            self.F.dat.data[:].reshape((-1,2))[:,0] -= α*Jn
            self.F.dat.data[:].reshape((-1,2))[:,1] += α*Jn

            # (2) Error
            iter +=1
            if np.linalg.norm(Jn) == 0:
                error = 0.
            else:    
                error = np.linalg.norm(Jn - Jo,2)/np.linalg.norm(Jn,2)
            Jo    = Jn

            if iter > 10**2:
                raise ValueError('Slope limiter relaxation iterations exceeded threshold \n')
            
            slopes = self.F.dat.data[:].reshape((-1,2))[:,1] - self.F.dat.data[:].reshape((-1,2))[:,0]
            slope  = np.min(slopes)
            if abs(slope) < 1e-12:
                slope = 0.

            #print('Iteration i=%d'%iter,' error = ',error,'slope =',slope,'\n')
            # if iter%10 == 0:
            #     print('Iteration i=%d'%iter,' error = ',error,'slope =',slope,'\n')
            #     ptp.plot(function='CDF')

        # B) Remove remaining illegal discontinuities 
        Jn = jumps(self.F,F_0)
        self.F.dat.data[:].reshape((-1,2))[:,0] -= Jn
        self.F.dat.data[:].reshape((-1,2))[:,1] += Jn

        return None

    def cdf(self):

        """
        Construct the CDF F_Y(y) of the random function Y(x) by projecting from physical space into the probability space specified.

        Inputs:

        quadrature_degree int - order of the numerical quadrature scheme to use

        """

        # Define trial & test functions on V_F_hat
        u = TrialFunction(self.ptp.V_F_hat)
        v = TestFunction( self.ptp.V_F_hat)

        # Construct the linear & bilinear forms
        a = inner(u,v) * dx
        L = inner(self.indicator(),v) * dx(degree=self.quadrature_degree)

        # Solve for F_hat
        F_hat = Function(self.ptp.V_F_hat)
        solve(a == L,F_hat)  

        # Recover F_Y(y) in V_F
        self.F = Function(self.ptp.V_F)

        # Sort a linear function in ascending order 
        # this creates a DOF map which matches 
        # the extended mesh which are in ascending order
        y,  = SpatialCoordinate(self.ptp.m_y)
        ys  = assemble(interpolate(y,self.ptp.V_F))
        indx= np.argsort(ys.dat.data)

        # Pass F_hat into F
        self.F.dat.data[indx] = F_hat.dat.data[:]
        
        # Apply a slope limiter to F
        #self.slope_limiter()

        # Check CDF properties
        Surf_int = assemble(self.F*ds)
        if abs(Surf_int - 1) > 1e-02:
            print("Calculated F(+∞) - F(-∞) should equal 1, got %e. Check the domain of Ω_Y and the quadrature_degree specified."%Surf_int)
            #raise ValueError("Calculated F(+∞) - F(-∞) should equal 1, got %e. Check the domain of Ω_Y and the quadrature_degree specified."%Surf_int)

        return None;

    def compose(self,f,g):

        """
        Returns the composition of two functions 
            
            f o g(y) = f(g(y))

        at the quadrature points y_q of a quadrature mesh. 
        """

        mesh_g = g.function_space().mesh()
        mesh_f = f.function_space().mesh()

        V_fgE = FiniteElement(family="Quadrature",cell="interval",degree=self.quadrature_degree,quad_scheme='default')
        V_fg  = FunctionSpace(mesh=mesh_g,family=V_fgE)
        fg    = Function(V_fg)

        m = V_fg.mesh()
        W = VectorFunctionSpace(m, V_fg.ufl_element())
        y_vec = assemble(interpolate(m.coordinates, W))

        y_q  = [ [y_i,] for y_i in y_vec.dat.data[:]]
        vom  = VertexOnlyMesh(mesh_g, y_q)
        P0DG = FunctionSpace(vom, "DG", 0)
        g_vec= assemble(interpolate(g, P0DG))

        g_q   = [ [g_i,] for g_i in g_vec.dat.data[:]]
        vom   = VertexOnlyMesh(mesh_f, g_q)
        P0DG = FunctionSpace(vom, "DG", 0)
        f_vec = assemble(interpolate(f, P0DG))

        fg.dat.data[:] = f_vec.dat.data[:]

        return fg

    def qdf(self):
        """
        Construct the QDF (inverse CDF) Q_Y(p) of the random function Y(x) by inverting F_Y(y) = p
        """

        # (1) Construct the non-uniform domain Ω_p

        # Obtain dofs F_i = F(z_i) from the CDF
        F_i = self.F.dat.data[:] 

        # We extend Ω_p to include the endpoints 0,1
        # As F(y=0) ≠ 0 & F(y=1) ≠ 1 due to numerical error  
        p   = np.hstack(( [0],F_i,[1] ))  

        # Make a 1D mesh whose vertices are given by the p values
        layers   = len(p[1:] - p[:-1]);
        self.m_p = UnitIntervalMesh(ncells=layers);
        self.m_p.coordinates.dat.data[:] = p[:]

        # (2) Create a function Q(p) on this mesh
        self.V_Q = FunctionSpace(mesh=self.m_p,family=self.ptp.V_QE)
        self.Q   = Function(self.V_Q)

        # (3) Extract the mesh coordinates of the CDF
        m_y = self.ptp.V_F.mesh()
        W   = VectorFunctionSpace(m_y, self.ptp.V_F.ufl_element())
        y_m = assemble(interpolate(m_y.coordinates, W)).dat.data

        # Append the coordinates of the boundaries
        y_l = m_y.coordinates.dat.data[ 0] # left endpoint
        y_r = m_y.coordinates.dat.data[-1] # right endpoint
        y_i = np.hstack(( [y_l],y_m,[y_r] )) 

        # Assign Q(F_i) = y_i
        self.Q.dat.data[:] = y_i[:]

        return None;

    def pdf(self):

        """
        Construct the PDF f_Y(y) of the random function Y(x) by projecting f_Y(y) = ∂y F_Y(y)
        """

        # Define trial & test functions on V_f
        u = TrialFunction(self.ptp.V_f)
        v = TestFunction(self.ptp.V_f)

        # Construct the linear & bilinear forms
        a =  inner(u,v) * dx
        L = -inner(self.F,v.dx(0)) * dx  +  self.F*v*ds(2) - self.F*v*ds(1)

        # Solve for f
        self.f = Function(self.ptp.V_f)
        solve(a == L, self.f)

        # Check PDF properties
        PDF_int = assemble(self.f*dx)
        if abs(PDF_int - 1) > 1e-02:
            print("Calculated ∫ f(y) dy should equal 1, but got %e. Check the quadrature_degree used. "%PDF_int)
            #raise ValueError("Calculated ∫ f(y) dy should equal 1, but got %e. Check the quadrature_degree used. "%PDF_int)

        return None;

    def fit(self):

        self.cdf()
        self.qdf()
        self.pdf()

        return None;

    def plot(self,function='CDF'):
        
        """
        Visualise the CDF, QDF and PDF using the inbuilt plotting routines
        """

        import matplotlib.pyplot as plt
        from firedrake.pyplot import plot

        if function == 'CDF':
            try:
                Line2D_F = plot(self.F,num_sample_points=150)
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
        
        # else:
            
        #     fig, (ax1, ax2) = plt.subplots(1, 2)

        #     Line2D_F = plot(self.F,num_sample_points=50)
        #     print(Line2D_F)
        #     ax1.add_line(Line2D_F[0])
        #     ax1.set_title(r'CDF',fontsize=20)
        #     ax1.set_ylabel(r'$F_Y$',fontsize=20)
        #     ax1.set_xlabel(r'$y$',fontsize=20)

        #     Line2D_Q = plot(self.Q,num_sample_points=50)
        #     ax2.add_line(Line2D_Q[0])
        #     ax2.set_title(r'QDF',fontsize=20)
        #     ax2.set_ylabel(r'$Q_Y$',fontsize=20)
        #     ax2.set_xlabel(r'$p$',fontsize=20)
            
        #     plt.tight_layout()
        #     plt.show()
            
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


# Takes instances of the FEptp object & performs operations
if __name__ == "__main__":

    # %%
    print("Initialise")
    
    # %%
    #1D example

    # (a) Specify the domain size(s) & number of finite elements/bins 
    ptp   = FEptp(Omega_X = {'x1':(0,1)}, Omega_Y = {'Y':(0,1)}, N_elements=5)
    x1,_  = ptp.coords

    # (b) Projection Y(X) into probability space    
    den = ptp.fit(function_Y = x1**1.5, quadrature_degree=100)
    den.plot('QDF')

    # # %%
    # #1D example of trying to specify a piecewise constant function

    # # (a) Specify the domain size(s) & number of finite elements/bins 
    # ptp   = FEptp(Omega_X = {'x1':(0,1)}, Omega_Y = {'Y':(0,1)}, N_elements=10)
    # x1,_  = ptp.coords
    
    # https://fenics.readthedocs.io/projects/ufl/en/latest/manual/form_language.html

    # #expression = conditional( lt(x1,1/3),x1,0) #+ conditional( gt(x1,2/3),x1,0) 

    # expression = conditional( gt(x1,2/3),x1,10) 

    # # (b) Projection Y(X) into probability space    
    # ptp_0 = ptp.fit(function_Y = expression, quadrature_degree=100)
    # ptp_0.plot(function='CDF')

    # # %%
    # #2D example
    # ptp    = FEptp(Omega_X = {'x1':(0,1),'x2':(0,1)}, Omega_Y = {'Y':(0,2)}, N_elements=5)
    # x1,x2,_ = ptp.coords
    # ptp.fit(function_Y = x1 + x2, quadrature_degree=500)
    # ptp.plot()

    # # # Evaluate the CDF & PDF at points
    # # F_Y,f_Y,y_i  = ptp.evaluate(y = [0., 0.1, 0.2])
# %%
