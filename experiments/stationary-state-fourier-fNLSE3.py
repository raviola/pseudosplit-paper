"""
Cubic Fractional Nonlinear Schrödinger Equation (fNLSE3) - Station. state evol.

Calculate the evolution of a stationary state with different split-step
symplectic and affine integrators. Uses Fourier pseudospectral representation.
Store various metrics (Inf error, L_2 norm error, Hamiltonian, computing time)
"""

import numpy as np
import time
from scipy.optimize import newton_krylov

from basis import FourierBasis
from state import State

from models import NLSE3Model1D

# Model parameters (fNLSE3)
D = 1.0
gamma = -1.0
alphas = np.linspace(2.0, 1.0, 41)
solutions = []

dim = 2**15
radius = 100 * np.pi

# dim = 300
# scaling = 1
# Set the parameters of the NLSE3 standing wave solution
omega = -1.

def NLS_ground_state(x):
    return 1.0 / np.cosh(x)

first = True

f0 = FourierBasis('fb', dim, (-radius, radius))

for alpha in alphas:
    # Create instance of fNLSE3 model
    model = NLSE3Model1D(D=D, gamma=gamma, s=alpha/2)
    
    # Get the pseudospectral representation of soliton
    if first:
        u0 = State('u0', f0, u=NLS_ground_state)
        first = False
    else:
        u0 = State('u0', f0)
        u0.values = u0values

    A_op = model.get_A_operator(f0)
    B_op = model.get_B_operator(f0)
    
    # Nonlinear functional to obtain the ground state numerically
    def F(psi):
        
        return A_op(psi) + B_op(psi) - omega * psi

    x = u0.grid
    u0values = u0.values
    
    # Solve nonlinear system
    u0values = newton_krylov(F, u0values, f_tol=1e-12, method='lgmres',
                             verbose=True)
    
    # Reset initial state for next alpha (now it's a ground state)
    u0.values = u0values
    
    # Initial mass (L2 norm)
    M0 = u0.norm()
    
    # Initial value of Hamiltonian (energy)
    hamiltonian = model.get_hamiltonian(f0)
    H0 = hamiltonian(u0)
    
    solutions.append(u0)
    print(alpha, M0, H0)
    
