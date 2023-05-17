"""
Complex Ginzburg-Landau soliton evolution.

Calculate the evolution of a soliton with different split-step
symplectic and affine integrators. Uses Fourier pseudospectral representation.
Store various metrics (Inf error, L_2 norm error, computing time)
"""

import time
import numpy as np
from scipy.integrate import solve_ivp

from basis import FourierBasis
from state import State
from scheme import (Strang, Ruth, Neri, Yoshida6,
                     AffineSym2, AffineSym4, AffineSym6,
                     DOPRI5, DOPRI853)

from models import GeneralModel1D
from solver import Solver

# Model parameters (fCGLE5)
D = 1.0
alpha = 2.0
beta = 2.0
gamma = -1.0
lamb = (1 + 4 * beta**2)**0.5
delta = 0.0
epsilon = beta * (3 * lamb - 1) / (4 + 18 * beta**2)
nu = 0
mu = 0

d = (lamb - 1) / (2 * beta)
F = (d * lamb / (2 * epsilon))**0.5
G = 1.0
omega = -d * lamb**2 / (2 * beta) * G**2

header_str = f"""\
# =========================
# Model parameters (CGLE3)
# =========================
# D = {D}
# alpha = {alpha}
# beta = {beta}
# gamma = {gamma}
# delta = {delta}
# epsilon = {epsilon}
# nu = {nu}
# mu = {mu}
# =========================
"""

# Numerical method parameters
# Pseudospectral parameters (spatial discretization)
dimensions = (2**12,)
radius = (100*np.pi,)

# Splitting parameters (temporal evolution)
schemes = (Strang, Ruth, Neri, Yoshida6,
           AffineSym2, AffineSym4, AffineSym6,
           DOPRI5, DOPRI853)

# Time steps
dt_list = [2.5e-0, 1.0e-0,
           5.0e-1, 2.5e-1, 1.0e-1,
           5.0e-2, 2.5e-2, 1.0e-2,
           5.0e-3, 2.5e-3, 1.0e-3]

# Initial and final times for simulation
t0 = 0.0
tf = 10.0


# Define the soliton function (Akhmediev 1996)
def initial_state(x):
    """
    Soliton solution for the CGLE.

    Parameters
    ----------
    x : array-like
        Space coordinates.

    Returns
    -------
    array-like
        Soliton values at the given space coordinates.

    """
    return G*F/np.cosh(G*x)*np.exp(1.0j*d*np.log(G*F/np.cosh(G*x)))


# Time stamp for file identification
time_stamp = time.localtime()
time_stamp_str = time.strftime("%Y-%m-%d-%H:%M:%S", time_stamp)

# Column names for output file
column_names = ("basis,N,scaling,"
                "method,delta_t,T,"
                "err_inf,err_2,comp_time,nfev,nfev2\n")

# Format string for screen output
screen_template = ("{0:8s} {1:4d} {2:e} "
                   "{3:8s} {4:e} {5:e} "
                   "{6:e} {7:e} {8:e} {9:6d} {10:6d}")

# Format string for file output
file_template = ("{0:8s},{1:4d},{2:e},"
                 "{3:8s},{4:e},{5:e},"
                 "{6:e},{7:e},{8:e},{9:6d},{10:6d}\n")


with open("output/CGLE3/fourier/CGLE3_soliton_fourier_2048-r_50.0-alpha_" + str(alpha) + "-"
          + time_stamp_str + ".csv", "w") as fout:

    # Write file headers
    fout.write(header_str)
    # fout.write("# Initial time: " + time_stamp_str + "\n")
    fout.write(column_names)

    # Create instance of fCGLE5 model
    model = GeneralModel1D(D=D, beta=beta, s=alpha/2, delta=delta,
                           gamma=gamma, epsilon=epsilon, nu=nu, mu=mu)

    # Iterate over parameters space
    for dim in dimensions:
        for r in radius:
            # Get the pseudospectral representation of the initial state
            b0 = FourierBasis('hb', dim, (-r, r))
            b_name = b0.name_prefix
            u0 = State('u0', b0, u=initial_state)

            for dt in dt_list:

                for Scheme in schemes:
                    # Select the split-step scheme and create the solver
                    scheme = Scheme()
                    solver = Solver(model, scheme)

                    # Times for evaluation of numerical solution
                    t_eval = np.linspace(t0, tf, int((tf-t0) / dt)+1)

                    # Calculate the reference solution with DOP853
                    # RHS = model.get_RHS(b0)

                    # def wrapped_RHS(t, y):
                    #     """
                    #     Wrap the RHS to adapt it to the scipy solver.

                    #     Parameters
                    #     ----------
                    #     t : float
                    #         Time.
                    #     y : array
                    #         Initial value.

                    #     Returns
                    #     -------
                    #     array
                    #         Final value.

                    #     """
                    #     return -1j * RHS(y)

                    x = u0.grid
                    # ref = solve_ivp(wrapped_RHS, (t0, tf),
                    #                 initial_state(x).astype(np.complex128),
                    #                 'DOP853', t_eval,
                    #                 rtol=2.25e-14, atol=1e-16)

                    # # Reference solution (last value of DOPRI853 integrator)
                    # ref_sol = ref.y[:, -1]

                    # A trajectory is a list of tuples (time, state)
                    trajectory = [(t0, u0)]

                    # Start the solver
                    solver.start(u0, t0, tf)
                    # Time at the beginning of calculations
                    comp_time_init = time.time()

                    # Evolution of the fCGLE5 dissipative soliton
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
                    ref_sol = u0.values * np.exp(-1j * omega * t)
                    num_sol = trajectory[-1][1].values

                    # Calculate metrics (errors and computational cost)
                    err_inf = np.amax(np.abs(ref_sol - num_sol))
                    err_2 = np.sqrt(np.sum((np.abs(num_sol - ref_sol))**2))

                    nfev_low = int(scheme.P_A.nfev * scheme.low_factor) if scheme.is_splitting else 1
                    nfev_high = int(scheme.P_A.nfev * scheme.high_factor) if scheme.is_splitting else 1
                    # Output to screen
                    print(screen_template.format(b_name, dim, r,
                                                 scheme.name, dt, tf,
                                                 err_inf, err_2,
                                                 comp_time, nfev_low, nfev_high))
                    # Output to CSV file
                    fout.write(file_template.format(b_name, dim, r,
                                                    scheme.name, dt, tf,
                                                    err_inf, err_2,
                                                    comp_time, nfev_low, nfev_high))
