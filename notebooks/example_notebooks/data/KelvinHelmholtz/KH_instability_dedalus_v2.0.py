import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# Parameters
d = 1/28  # shear-layer thickness
c = 0.05  # Perturbation amplitude
Lx, Lz = 4*np.pi, np.pi
Nx, Nz = 1024, 512
Re, Pr, Ri = 10**4, 1, 0.1
dealias = 3/2
stop_sim_time = 25
timestepper = d3.RK443
max_timestep = 0.005
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(-Lz, Lz), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis, zbasis))
b = dist.Field(name='b', bases=(xbasis, zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis, zbasis))
tau_p = dist.Field(name='tau_p')
tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

# Substitutions
mu = 1/(Re*Pr)
nu = 1/Re
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1)  # First-order reduction
grad_b = d3.grad(b) + ez*lift(tau_b1)  # First-order reduction


# Defining stress-free boundary conditions
u_x = u @ ex
u_z = u @ ez
dzu_x = d3.Differentiate(u_x, coords['z'])

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(b) - mu*div(grad_b) + lift(tau_b2)                     = -u@grad(b)")
problem.add_equation("dt(u) - nu*div(grad_u) + lift(tau_u2) + grad(p) - Ri*b*ez = -u@grad(u)")
problem.add_equation("integ(p) = 0")  # Pressure gauge

problem.add_equation("b(z=-Lz) = -1")
problem.add_equation("b(z= Lz) =  1")

problem.add_equation("dzu_x(z=-Lz) = 0")
problem.add_equation("dzu_x(z= Lz) = 0")
problem.add_equation("u_z(z=-Lz) = 0")
problem.add_equation("u_z(z= Lz) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
b['g'] = np.tanh(z/d)
u['g'][0] = np.tanh(1.1*z/d)
u['g'][1] = 0.

k1 = 8*np.pi/Lx
k2 = 20*np.pi/Lx
u['g'][0] += c*np.exp(-z**2/d**2)*(-k1*np.sin(k1*x)-k2*np.sin(k2*x))
u['g'][1] -= c*(-2*z/d**2)*np.exp(-z**2/d**2)*(np.cos(k1*x)+np.cos(k2*x))

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=1, max_writes=50)
snapshots.add_task(b, name='buoyancy', scales=2)
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity', scales=2)

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, 
             threshold=0.05, max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u), name='Ke')

# Main loop
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Ke = flow.max('Ke')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Ke)=%f' %(solver.iteration, solver.sim_time, timestep, max_Ke))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()