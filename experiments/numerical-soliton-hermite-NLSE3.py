#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cubic Nonlinear Schrödinger Equation (NLSE3) -  Soliton evolution.

Calculate the evolution of a soliton with different split-step
symplectic and affine integrators. Uses Hermite pseudospectral representation.
Store various metrics (max error, L_2-norm error, hamiltonian, computing time)
"""

import numpy as np
import time

from basis import HermiteBasis
from state import State
from scheme import (Strang, Ruth, Neri, Yoshida6, DOPRI853, DOPRI5,
                     AffineSym2, AffineSym4, AffineSym6, AffineSym8)

from models import NLSE3Model1D
from solver import Solver

# Model parameters (NLSE3)
D = 1.0
gamma = -1.0

header_str = f"""\
# =========================
# Model parameters (NLS3)
# =========================
# D = {D}
# gamma = {gamma}
# =========================
"""


# Numerical method parameter space
# Pseudospectral parameters (spatial discretization)
dimensions = (10, 20, 40, 80, 160, 320, 640)
# dimensions = (300,)
scalings = (1.25,)  # (0.5, 0.75, 1.0, 1.25, 1.5,1.75, 2.0, 2.5, 3.0, 3.5, 4.0)

# Splitting parameters (temporal evolution)
schemes = (Strang, Ruth, Neri, Yoshida6,
           AffineSym2, AffineSym4, AffineSym6) #, AffineSym8,
           #DOPRI5, DOPRI853)

# dt_list = [2.5e-0, 1.0e-0,
#            5.0e-1, 2.5e-1, 1.0e-1,
#            5.0e-2, 2.5e-2, 1.0e-2,
#            5.0e-3, 2.5e-3, 1.0e-3]

dt_list = [0.025, ]
# Initial and final times for simulation
t0 = 0.0
tf = 10.0

# Set the parameters of the NLSE3 soliton solution
c = 0.5
eta = 1.0
omega = 0.5*(c**2 - eta**2)


# Define the soliton function
def soliton(x):
    """
    Soliton solution for the NLSE3 (focusing case).

    Parameters
    ----------
    x : array-like
        Space coordinates.

    Returns
    -------
    array-like
        Soliton values at the given space coordinates.

    """
    return eta / np.cosh(eta*(x)) * np.exp(1.0j*(c*x))


# Time stamp for file identification
time_stamp = time.localtime()
time_stamp_str = time.strftime("%Y-%m-%d-%H:%M:%S", time_stamp)

# Column names for output file
column_names = ("basis,N,scaling,"
                "method,delta_t,T,"
                "err_inf,err_2,err_M,err_H,comp_time,nfev,nfev2\n")

# Format string for screen output
screen_template = ("{0:8s} {1:4d} {2:e} "
                   "{3:8s} {4:e} {5:e} "
                   "{6:e} {7:e} {8:e} {9:e} {10:e} {11:6d} {12:6d}")

# Format string for file output
file_template = ("{0:8s},{1:4d},{2:e},"
                 "{3:8s},{4:e},{5:e},"
                 "{6:e},{7:e},{8:e},{9:e},{10:e},{11:6d},{12:6d}\n")


with open("output/NLSE3/hermite/soliton-hermite_300_s_1.25_c_0.5_dt_0.025-"
          + time_stamp_str + ".csv", "w") as fout:
    # Write file headers
    fout.write(header_str)
    # fout.write("# Initial time: " + time_stamp_str + "\n")
    fout.write(column_names)

    # Create instance of fNLSE3 model
    model = NLSE3Model1D(D=D, gamma=gamma)

    for dim in dimensions:
        for scaling in scalings:
            # Get the pseudospectral representation of the initial state
            b0 = HermiteBasis('hb', dim, (0., scaling))
            b_name = b0.name_prefix
            u0 = State('u0', b0, u=soliton)

            # Initial mass (L2 norm)
            M0 = u0.norm()

            # Initial value of Hamiltonian (energy)
            hamiltonian = model.get_hamiltonian(b0)
            H0 = hamiltonian(u0)
            print(H0)

            for dt in dt_list:
                for Scheme in schemes:

                    # Select the split-step scheme and create the solver
                    scheme = Scheme()
                    solver = Solver(model, scheme)

                    # A trajectory is a list of tuples (time, state)
                    trajectory = [(t0, u0)]
                    # Start the solver
                    solver.start(u0, t0, tf)
                    
                    # Times for evaluation of numerical solution
                    t_eval = np.linspace(t0, tf, int((tf-t0) / dt)+1)
                    
                    # Time at the beginning of calculations
                    comp_time_init = time.time()
                    
                    i = 0
                    while solver.active:
                        u = solver.step(t_eval[i+1] - t_eval[i])
                        t = solver.sim_time
                        trajectory.append((t, u))
                        i += 1

                    # Time at the end of calculations
                    comp_time_final = time.time()
                    # Time elapsed during calculations
                    comp_time = comp_time_final - comp_time_init

                    x = u0.grid
                    # Theoretical final state
                    ref_sol = eta / (
                        np.cosh(eta*(x-c*t))) * np.exp(1.0j*(c*x-omega*t))
                    # Numerical final state
                    num_sol = trajectory[-1][1].values

                    # Calculate metrics (errors and computational cost)
                    err_M = np.abs(M0 - u.norm()) / M0
                    err_H = np.abs((H0 - hamiltonian(u)) / H0)
                    err_inf = np.amax(np.abs(num_sol - ref_sol))
                    err_2 = np.sqrt(np.sum((np.abs(num_sol - ref_sol))**2))

                    # TODO: calculate cost for Runge-Kutta schemes
                    nfev_low = int(scheme.P_A.nfev * scheme.low_factor) if scheme.is_splitting else 1
                    nfev_high = int(scheme.P_A.nfev * scheme.high_factor) if scheme.is_splitting else 1
                    # Output to screen
                    print(screen_template.format(b_name, dim, scaling,
                                                 scheme.name, dt, tf,
                                                 err_inf, err_2,
                                                 err_M, err_H,
                                                 comp_time,
                                                 nfev_low,
                                                 nfev_high))
                    # Output to CSV file
                    fout.write(file_template.format(b_name, dim, scaling,
                                                    scheme.name, dt, tf,
                                                    err_inf, err_2,
                                                    err_M, err_H,
                                                    comp_time,
                                                    nfev_low,
                                                    nfev_high))
