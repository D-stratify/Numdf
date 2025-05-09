"""
Generate the figures for the NumDF homepage.
"""

from numdf import Ptp
import numpy as np
from firedrake import *
import matplotlib.pyplot as plt 
import h5py
from scipy.interpolate import RegularGridInterpolator
from matplotlib import rc
rc('text', usetex=True)


fig, ax = plt.subplots(ncols=4, figsize=(14, 3), layout='constrained')

# A) Layered-Stratification
# ~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~


def Y_numerical(X, alpha=1000):
    """Return Y(X)."""
    x1, x2 = X[:, 0], X[:, 1]
    return .25*( np.tanh(alpha*(x2-.5) ) + np.tanh(alpha*(x2 +.5) ) + np.tanh(alpha*(x2-.25) ) + np.tanh(alpha*(x2 +.25) ) )


# A.1 Create the space plot
x = np.linspace(-1, 1, 10**2)
z = np.linspace(-1, 1, 10**2)
[x1, x2] = np.meshgrid(x, z, indexing='ij')

X = np.vstack((x1.flatten(), x2.flatten())).T
Y = Y_numerical(X).reshape(x2.shape)[:-1, :-1]

cmap = plt.get_cmap('RdBu')
im = ax[0].pcolormesh(x1, x2, Y, cmap=cmap)
ax[0].set_title(r'$Y(X_1,X_2)$', fontsize=20)
ax[0].set_xlabel(r'$X_1$', fontsize=20)
ax[0].set_ylabel(r'$X_2$', fontsize=20)

# A.2 Create the CDF
ptp = Ptp(Omega_X={'x1': (-1, 1)}, Omega_Y={'Y': (-1, 1)}, n_elements=50)
x2 = ptp.x_coords()
alpha = 1000
B = ( tanh(alpha*(x2 - 1/2)) + tanh(alpha*(x2 + 1/2)) + tanh(alpha*(x2 - 1/4)) + tanh(alpha*(x2 + 1/4)) )/4
density_B = ptp.fit(Y=B, quadrature_degree=500)

y = np.linspace(-1, 1, 10**2)
F = np.asarray(density_B.cdf.at(y))
ax[1].plot(y, F, 'b', linewidth=2)

ax[1].set_xlabel(r'$y$', fontsize=20)
ax[1].set_ylim([0, 1])
ax[1].set_xlim([-1, 1])
ax[1].grid()
ax[1].set_ylabel(r'$F(y)$', fontsize=20)


# B) Kelvin-Helmholtz
# ~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~


def Y_numerical(X, time=20):
    """Return Y(X)."""
    f = h5py.File('/home/pmannix/Finite_Element/NumDF/notebooks/example_notebooks/data/KelvinHelmholtz/snapshots_s1.h5','r')
    Y = f['tasks/buoyancy'][time, ...]
    x = f['tasks/buoyancy'].dims[1][0][:]
    z = f['tasks/buoyancy'].dims[2][0][:]
    f.close()

    return RegularGridInterpolator((x, z), Y, bounds_error=False)(X)


# B.1 Create the KH space plot
dx1, dx2 = 0.01, 0.01
x2, x1 = np.mgrid[slice(-np.pi, np.pi + dx1, dx1), slice(0, 4*np.pi + dx2, dx2)]
X = np.vstack((x1.flatten(), x2.flatten())).T
Y = Y_numerical(X).reshape(x2.shape)[:-1, :-1]

cmap = plt.get_cmap('RdBu')
im = ax[2].pcolormesh(x1, x2, Y, cmap=cmap)
ax[2].set_title(r'$Y(X_1,X_2)$', fontsize=20)
ax[2].set_xlabel(r'$X_1$', fontsize=20)
ax[2].set_ylabel(r'$X_2$', fontsize=20)


# B.2 Create the plot of the PDF and CDF
ptp = Ptp(Omega_X={'x1': (0, 4*np.pi), 'x2': (-np.pi, np.pi)}, Omega_Y={'Y': (-1, 1)}, n_elements=50)

Y = lambda X: Y_numerical(X)
density = ptp.fit(Y, quadrature_degree=500)

y = np.linspace(-1, 1, 10**2)
F = np.asarray(density.cdf.at(y))
ax[3].plot(y, F, 'b', linewidth=2)

ax[3].set_xlabel(r'$y$', fontsize=20)
ax[3].set_ylim([0, 1])
ax[3].set_xlim([-1, 1])
ax[3].grid()
ax[3].set_ylabel(r'$F(y)$', fontsize=20)


# Save figure
# ~~~~~~~~~~ # ~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~

line = plt.Line2D((.495,.495),(.05,.95), color="0.8", linewidth=3)
fig.add_artist(line)

for ax_i in ax:
    ax_i.set_box_aspect(0.6)
    ax_i.tick_params(axis='both', labelsize=16)

fig.savefig(fname='Homepage_Examples.png', dpi=200)
plt.close(fig)
