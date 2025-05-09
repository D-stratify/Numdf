"""
Generate the figures for the Kelvin-Helmholtz example presented in the paper.
"""

from numdf import Ptp
import numpy as np
from firedrake import *
import matplotlib.pyplot as plt 
import h5py
from scipy.interpolate import RegularGridInterpolator
from matplotlib import rc
rc('text', usetex=True)


# 1) Evaluate the numerical simulation at a set of quadrature points
Times = [16, 20, 24]
quadrature_degree = 500
n_elements = 88


def Y_numerical(X, time=2):
    """Return Y(X)."""

    f = h5py.File('/home/pmannix/Finite_Element/NumDF/notebooks/example_notebooks/data/KelvinHelmholtz/snapshots_s1.h5','r')
    Y = f['tasks/buoyancy'][time, ...]
    x = f['tasks/buoyancy'].dims[1][0][:]
    z = f['tasks/buoyancy'].dims[2][0][:]
    f.close()

    return RegularGridInterpolator((x, z), Y, bounds_error=False)(X)


fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 5.25), layout='constrained')
ax_space = ax[0, :]
ax_cdf = ax[1, :]

# 2) Create a space time plot of the simulation
dx1, dx2 = 0.01, 0.01
x2, x1 = np.mgrid[slice(-np.pi, np.pi + dx1, dx1), slice(0, 4*np.pi + dx2, dx2)]
X = np.vstack((x1.flatten(), x2.flatten())).T

for i, ax0, time in zip([0, 1, 2], ax_space, Times):

    Y = Y_numerical(X, time).reshape(x2.shape)[:-1, :-1]

    cmap = plt.get_cmap('RdBu_r')
    im = ax0.pcolormesh(x1, x2, Y, cmap=cmap)
    ax0.set_title(r'$T=%d$'%time,fontsize=20)
    ax0.set_xlabel(r'$X_1$',fontsize=20)
    ax0.set_box_aspect(0.5)
    if i == 0:
        ax0.set_ylabel(r'$X_2$', fontsize=20)
    else:
        ax0.set_yticklabels([])
    ax0.tick_params(axis='both', labelsize=16)

# 3) Create the time evolution of the PDF and CDF
ptp = Ptp(Omega_X={'x1': (0, 4*np.pi), 'x2': (-np.pi, np.pi)}, Omega_Y={'Y': (-1.1, 1.1)}, n_elements=n_elements)
y = np.linspace(-1.1, 1.1, 10**3)

for i, ax0, time in zip([0, 1, 2], ax_cdf, Times):

    print('Evaluating function i=%d/3 \n' % (i+1))

    Y = lambda X: Y_numerical(X, time=time)
    density = ptp.fit(Y, quadrature_degree=quadrature_degree)

    F = np.asarray(density.cdf.at(y))
    ax0.plot(y, F, 'k', linewidth=1)

    fc = np.asarray(density.pdf['fc'].at(y))
    #ax0.plot(y, fc, 'r', linewidth=2)
    ax0.fill_between(x=y, y1=fc, color="r", alpha=0.25)

    # Plot the Dirac measures
    m_y = density.pdf["fs"].function_space().mesh()
    loc = m_y.coordinates.dat.data[:]
    jump = density.pdf["fs"].dat.data[:]

    for j_i, y_i in zip(jump, loc):
        ax0.plot(y_i*np.ones(50), j_i*np.linspace(0, 1, 50), 'b--', linewidth=2)
        if j_i > 5e-03: ax0.plot(y_i, j_i, marker = 'o', ms = 5, mfc ='w', mec = 'b')

    ax0.set_xlabel(r'$b$', fontsize=20)
    ax0.set_ylim([0, 1.5])
    ax0.set_xlim([-1.1, 1.1])
    ax0.set_box_aspect(0.5)
    ax0.grid()
    #ax0.legend(loc='upper left', fontsize=20)
    if i == 0:
        ax0.set_ylabel(r'$\mathsf{F}_B, \; \mathsf{f}_B$', fontsize=20)
    else:
        ax0.set_yticklabels([])
    ax0.tick_params(axis='both', labelsize=16)


# 4) Calculate the APE as a function of time
def tpe(time=1):
    """Return the TPE at time t."""

    f = h5py.File('/home/pmannix/Finite_Element/NumDF/notebooks/example_notebooks/data/KelvinHelmholtz/snapshots_s1.h5','r')
    B = f['tasks/buoyancy'][time, ...]
    x = f['tasks/buoyancy'].dims[1][0][:]
    z = f['tasks/buoyancy'].dims[2][0][:]
    f.close()

    Z = np.outer(np.ones(len(x)), z)
    V = 8*(np.pi**2)

    return -(1/V)*np.trapezoid(y=np.trapezoid(y=B*Z, x=x, axis=0), x=z)


def bpe(time=1):
    """Return the BPE at time  t."""

    # Z - geo-potential
    ptp_Z = Ptp(Omega_X={'x1': (0, 4*np.pi), 'x2': (-np.pi, np.pi)}, Omega_Y={'Y': (-np.pi, np.pi)}, n_elements=5)
    x1, x2 = ptp_Z.x_coords()
    density_Z = ptp_Z.fit(Y=x2, quadrature_degree=quadrature_degree)
    
    # B - buoyancy
    ptp_B = Ptp(Omega_X={'x1': (0, 4*np.pi), 'x2': (-np.pi, np.pi)}, Omega_Y={'Y': (-1.1, 1.1)}, n_elements=n_elements)
    B = lambda X: Y_numerical(X, time)
    density_B = ptp_B.fit(Y=B, quadrature_degree=quadrature_degree)

    # Reference height
    F_Z = density_Z.cdf
    Q_B = density_B.qdf
    beta_star = density_B.compose(Q_B, F_Z, quadrature_degree=quadrature_degree)

    # bpe integral
    z = density_Z.y
    return density_Z(beta_star*z)


print('Kelvin Helmholtz energetics')
for ax0, t in zip(ax_space, Times):

    TPE = tpe(time=t)
    BPE = bpe(time=t)
    APE = BPE + TPE
    print('BPE(t = %d) = %6.6f   ' % (t, BPE))
    print('TPE(t = %d) = %6.6f   ' % (t, TPE))
    print('APE(t = %d) = %6.6f \n' % (t, APE))

    ax0.set_title(r'$\mathrm{APE}(T=%d) = %1.4f$' % (t, APE), fontsize=20)

fig.savefig(fname='KH_plot.png', dpi=400)
plt.close(fig)
