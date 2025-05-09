"""
Generate the figures for the Inverse CDF example presented in the paper.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from firedrake import *
from firedrake.__future__ import interpolate
from numdf import Ptp
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

def main():
    
    # Initialize the Ptp class with domain sizes and number of elements
    ptp = Ptp(Omega_X={'x1': (0, 1)}, Omega_Y={'Y': (0, 1)}, n_elements=3)

    # Define the function Y(x1) and fit the CDF and PDF
    x1 = ptp.x_coords()
    density = ptp.fit(Y=x1, quadrature_degree=10)

    # Plot the CDF
    #density.plot(function='CDF')

    # Obtain dofs F_i = F(z_i) from the CDF
    F_i = density.cdf.dat.data[:]

    # Extend Ω_p to include the endpoints 0 and 1
    p = np.hstack(([0], F_i, [1]))

    # Create a 1D mesh with vertices given by the p values
    layers = len(p[1:] - p[:-1])
    m_p = UnitIntervalMesh(ncells=layers)
    m_p.coordinates.dat.data[:] = p[:]
    print('Ω_p cell boundaries = ', m_p.coordinates.dat.data[:])

    # Define the function Q(p) on this mesh
    V_Q = FunctionSpace(mesh=m_p, family="CG", degree=1)
    Q = Function(V_Q)

    # Extract the mesh coordinates of the CDF
    m_y = ptp.V_F.mesh()
    w = VectorFunctionSpace(m_y, ptp.V_F.ufl_element())
    y_m = assemble(interpolate(m_y.coordinates, w)).dat.data

    # Append the coordinates of the boundaries
    y_l = m_y.coordinates.dat.data[0]  # left endpoint
    y_r = m_y.coordinates.dat.data[-1]  # right endpoint
    y_i = np.hstack(([y_l], y_m, [y_r]))

    # Assign Q(F_i) = y_i
    Q.dat.data[:] = y_i[:]

    # Make a fancy plot the CDF and the QDF

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout='constrained')

    # Plot the CDF
    #y_i = m_y.coordinates.dat.data[:]
    y_i = [0, 1/3, 1/3, 2/3, 2/3, 1]
    F_i = density.cdf.dat.data[:]
    axes[0].plot(y_i, F_i, 'bo', markerfacecolor='w', label="CDF", markersize=10)
    
    for i in range(len(y_i)-1):
        axes[0].plot([0, y_i[i+1]], [F_i[i+1], F_i[i+1]], 'k--', linewidth=2)
        #axes[0].plot([y_i[i+1], y_i[i+1]], [F_i[i], F_i[i+1]], 'k', linewidth=2)

    eps = 1e-4
    y = np.linspace(0, 1/3 - eps, 100)
    F = np.asarray(density.cdf.at(y))
    axes[0].plot(y, F, 'k', linewidth=2)

    y = np.linspace(1/3 + eps, 2/3 - eps, 100)
    F = np.asarray(density.cdf.at(y))
    axes[0].plot(y, F, 'k', linewidth=2)

    y = np.linspace(2/3 + eps, 1, 100)
    F = np.asarray(density.cdf.at(y))
    axes[0].plot(y, F, 'k', linewidth=2)
    
    axes[0].set_xlabel(r"$y$", fontsize=20)
    axes[0].set_ylabel(r"$\mathsf{F}_Y(y)$", fontsize=20)
    #axes[0].grid(True)
    axes[0].set_ylim([-0.01, 1.01])
    axes[0].set_xlim([-0.01, 1.01])


    # Plot the QDF
    t_i = m_p.coordinates.dat.data[:]
    Q_i = Q.dat.data[:]
    axes[1].plot(t_i, Q_i, 'bo', markerfacecolor='w', label="QDF", markersize=10)

    for i in range(len(t_i)-1):
        axes[1].plot([t_i[i+1], t_i[i+1]], [0, Q_i[i+1]], 'k--', linewidth=2)
        #axes[0].plot([y_i[i+1], y_i[i+1]], [F_i[i], F_i[i+1]], 'k', linewidth=2)

    t = np.linspace(0, 1, 1000)
    Q = np.asarray(density.qdf.at(t))
    axes[1].plot(t, Q, 'k',  linewidth=2)

    axes[1].set_xlabel(r"$t$", fontsize=20)
    axes[1].set_ylabel(r"$\mathsf{F}^{-1}_Y(t)$", fontsize=20)
    #axes[1].grid(True)
    axes[1].set_ylim([-0.01, 1.01])
    axes[1].set_xlim([-0.01, 1.01])

    #axes[0].fill_between(x=y, y1=F, y2=1, color="r", alpha=0.25)
    axes[1].fill_between(x=t, y1=Q, color="r", alpha=0.25)

    axes[0].tick_params(axis='both', labelsize=16)
    axes[1].tick_params(axis='both', labelsize=16)

    # Adjust layout and show the plot
    fig.savefig(fname='QDF_plot.png', dpi=400)
    plt.show()

if __name__ == "__main__":
    main()