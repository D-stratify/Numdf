"""
class which given a user provided "function" over a physical "domain"

    Y(x) where x in Ω

returns the "CDF" & "PDF"

    F_Y(y), f_Y(y)

over its corresponding probability space  ( Ω_Y, Y, I) using a finite 
element discretisation consisting of n_bins
"""

from firedrake import *


class PDF_Projector(object):

    def __init__(self, function,domain,n_bins = 10,func_space_CDF = {"family":"DG","degree":1},func_space_PDF= {"family":"CG","degree":1}):

        # Physical space
        self.Y = function
        
        self.X_dim = len(domain)
        
        if self.X_dim == 1:
            self.x1 = domain['x1']
        elif self.X_dim == 2:
            self.x1 = domain['x1']
            self.x2 = domain['x2']
        else:
            raise ValueError('The domain Ω must be 1D or 2D \n')

        # Probability space
        self.func_space_CDF = func_space_CDF
        self.func_space_PDF = func_space_PDF
        
        self.n = n_bins
        self.F = None
        self.f = None

        # Function spaces
        self.mesh = None

        return None;

    def Mesh(self):
        
        """
        Construct the extruded mesh 
            Ω x Ω_Y
        the physical space times the event space
        """

        # x-direction
        if   self.X_dim == 1:
            cell_type = "interval";  
            mesh_x    = IntervalMesh(ncells=1,length_or_left=self.x1[0],right=self.x1[1])
        elif self.X_dim == 2: 
            cell_type = "triangle";
            mesh_x    = RectangleMesh(nx=1,ny=1,Lx=self.x1[1],Ly=self.x2[1],originX=self.x1[0],originY=self.x2[0])

        # Add y-direction
        self.mesh = ExtrudedMesh(mesh_x, layers= self.n) 

        # # Set the function-space \hat{V}_F
        # R   = FiniteElement(family="DG",cell=cell_type ,degree=0)
        # V_F = FiniteElement(family="DG",cell="interval",degree=1)

        # elt_RVF = TensorProductElement(R,V_F)
        # V_hat_F = FunctionSpace(mesh,elt_RVF)

        return None;

    def CDF(self):

        return None;

    def PDF(self):

        return None;


if __name__ == "__main__":

    domain = {'x1':(-6,6),'x2':(-6,6)}
    print("Testing")