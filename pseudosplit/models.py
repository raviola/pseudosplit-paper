"""
Models for evolution equations.

@author: lisandro
"""
import numpy as np
from numpy import pi, abs, exp, log
from scipy.linalg import expm
from scipy.fft import fft, ifft, fftfreq
from hermite import hermite_nodes, hermite_matrix
from basis import HermiteBasis, FourierBasis

from scipy.integrate import solve_ivp

class GeneralModel1D():
    """General model for 1D NLS/CGL/GP-like equations."""

    def __init__(self,
                 D=1, beta=0, s=1, delta=0,
                 gamma=-1, epsilon=0,
                 nu=0, mu=0,
                 V=None
                 ):

        # Symbol for the Fourier multiplier A
        # including dispersion (D), diffusion (beta),
        # fractional exponent for Laplacian (s), linear gain/loss (delta),
        # cubic self-phase modulation (gamma), nonlinear cubic gain (epsilon),
        # quintic self-phase modulation (nu), nonlinear quintic gain (mu)

        self.A_symbol = lambda k: (D/2 - 1j*beta) * abs(k)**(2*s) + 1j * delta

        # Parameters for B (nonlinear terms and linear potential)
        self.D = D
        self.beta = beta
        self.s = s
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon
        self.nu = nu
        self.mu = mu
        self.V = V

    def get_A_operator(self, basis):

        if isinstance(basis, FourierBasis):
            k = 2 * pi * fftfreq(basis.dim,
                                 basis.period / basis.dim)
            A_k = self.A_symbol(k)

            def A_operator(u0):
                u0_til = 1 / basis.dim * fft(u0)
                A_u0_til = A_k * u0_til
                u = basis.dim * ifft(A_u0_til)
                return u

        elif isinstance(basis, HermiteBasis):
            # Compute the Gauss-Hermite quadrature with n = dim + 1 points
            # for exact integration.

            A_operator_matrix = self.get_A_matrix(basis)

            def A_operator(u0):
                return A_operator_matrix @ u0

        return A_operator

    def get_B_operator(self, basis):
        x = basis.get_grid()
        V_x = self.V(x) if self.V is not None else 0

        def B_operator(u0):
            return u0 * (V_x + np.abs(u0)**2 *
                         ((self.gamma + 1j*self.epsilon) +
                          (-self.nu + 1j*self.mu) * np.abs(u0**2)))
        return B_operator


    def get_A_matrix(self, basis, spectral=False):

        if isinstance(basis, HermiteBasis):
            # Compute the Gauss-Hermite quadrature with n = dim + 1 points
            # for exact integration.
            n = basis.dim + 1
            k, w = hermite_nodes(n, weights=True)
            A_k = self.A_symbol(k / basis.scaling)
            phi_quad = hermite_matrix(n-1, k)
            A_matrix = (w * A_k * phi_quad) @ phi_quad.T
            for row in range(n-1):
                for col in range(n-1):
                    A_matrix[row, col] = (-1j)**(row-col) * A_matrix[row, col]

            if spectral:
                return A_matrix.T
            else:
                phi = hermite_matrix(n-1)
                c2h = phi * basis.weights
                h2c = phi.T

                return h2c @ A_matrix.T @ c2h
    
    def get_RHS(self, basis):
        A_func = self.get_A_operator(basis)
        B_func = self.get_B_operator(basis)
        
        def RHS(u0):
            RHS.nfev += 1
            return A_func(u0) + B_func(u0)
        
        RHS.nfev = 0
        
        return RHS
        

    def get_propagators(self, basis):
        """
        Return the partial propagators A / B for the model according to basis.

        Parameters
        ----------
        basis : Basis
            An instance of a Basis class.

        Returns
        -------
        (P_A, P_B) : tuple of functions (partial propagators in given basis).

        """
        # If given, evaluate the potential at grid points
        x = basis.get_grid()
        V_x = self.V(x) if self.V is not None else 0

        if isinstance(basis, FourierBasis):
            # For Fourier basis, evaluate the symbol at discrete wavenumbers
            k = 2 * pi * fftfreq(basis.dim,
                                 basis.period / basis.dim)
            A_k = self.A_symbol(k)

            # Propagator for A in Fourier basis
            def P_A(u0, dt):
                propagator = exp(-1j * A_k * dt)
                u0_til = 1 / basis.dim * fft(u0)
                u0_til_evol = propagator * u0_til
                u = basis.dim * ifft(u0_til_evol)
                P_A.nfev += 1
                return u

        elif isinstance(basis, HermiteBasis):
            # Compute the Gauss-Hermite quadrature with n = dim + 1 points
            # for exact integration.
            # n = basis.dim + 1
            # kn, wn = hermite_nodes(n, weights=True)
            # A_kn = self.A_symbol(kn / basis.scaling)
            # phi_quad = hermite_matrix(n-1, kn)
            # A_matrix = (wn * A_kn * phi_quad) @ phi_quad.T
            # for row in range(n-1):
            #     for col in range(n-1):
            #         A_matrix[row, col] = (-1j)**(row-col) * A_matrix[row, col]

            # C = -1j * A_matrix.T
            phi = hermite_matrix(basis.dim)
            c2h = phi * basis.weights
            h2c = phi.T
            C = -1j * self.get_A_matrix(basis, spectral=True)
            # A_matrix_space = phi.T @ A_matrix.T @ phi * basis.weights
            dt_list = []
            prop_list = []

            def P_A(u0, dt):
                if dt not in dt_list:
                # if (dt not in P_A.dt):
                    # P_A.dt.append(dt)
                    dt_list.append(dt)
                    expCdt = expm(C * dt)
                    # P_A.propagator.append(h2c @ (expCdt @ c2h))
                    prop_list.append(h2c @ expCdt @ c2h)
                    # prop_list.append( expCdt)
                # i = P_A.dt.index(dt)
                i = dt_list.index(dt)
                # u = P_A.propagator[i] @ u0
                u = prop_list[i] @ u0
                P_A.nfev += 1
                return u

            #P_A.dt = []
            #P_A.propagator = []
        P_A.nfev = 0

        if self.nu * self.mu == 0:
            # Model without quintic term
            if self.epsilon == 0:
                # Model without nonlinear gain
                def P_B(u0, dt):
                    u = u0 * np.exp(-1j * dt * (self.gamma * abs(u0)**2 + V_x))
                    P_B.nfev += 1
                    return u
            else:
                # Model with nonlinear gain
                def P_B(u0, dt):
                    u = u0 * exp(-1j * V_x * dt +
                                 0.5 * (1j*self.gamma / self.epsilon - 1) *
                                 log(1 - 2*self.epsilon * abs(u0)**2 * dt))
                    P_B.nfev += 1
                    return u
        else:
            # Cubic-quintic model
            # The nonlinear propagator is obtained numerically with DOPRI853
            B_op = self.get_B_operator(basis)

            def RHS(t, y):
                return -1j * B_op(y)

            def P_B(u0, dt):
                # print(u0.shape)
                sol = solve_ivp(RHS, (0., dt),
                                u0.astype(np.complex128),
                                'DOP853')
                P_B.nfev += 1

                return sol.y[:, -1]
        P_B.nfev = 0
        return P_A, P_B

class NLSE3Model1D(GeneralModel1D):
    def __init__(self, D=1, gamma=-1, s=1):
        super().__init__(D=D, gamma=gamma, s=s)
    
    def get_hamiltonian(self, basis):
        
        if isinstance(basis, HermiteBasis):
            K_matrix = self.get_K_matrix(basis)
            
            def H(u):
                u_j = u.coeffs
                K = u_j @ (K_matrix @ u_j.conj())
                V = 0.5 * self.gamma * u.norm(4)
                return (K + V).real
            return H
        elif isinstance(basis, FourierBasis):
            
            def H(u):
                u_k_til = u.coeffs
                k = 2 * pi * fftfreq(basis.dim,
                                     basis.period / basis.dim)
                K = 0.5 * np.sum(np.abs(k * u_k_til)**(2*self.s))
                V = 0.5 * self.gamma * u.norm(4)
                return (K + V).real
            return H

    def get_K_matrix(self, basis):
        if isinstance(basis, HermiteBasis):
            K_matrix = basis.scaling * self.get_A_matrix(basis, spectral=True)
        elif isinstance(basis, FourierBasis):
            raise NotImplementedError
        return K_matrix

class FisherModel1D:
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        self.A_symbol = lambda k: 1j * (-k**2*alpha + 1 * beta)  # Symbol for Laplacian + identity
    
    def get_A_operator(self, basis):

        if isinstance(basis, FourierBasis):
            k = 2 * pi * fftfreq(basis.dim,
                                 basis.period / basis.dim)
            A_k = self.A_symbol(k)

            def A_operator(u0):
                u0_til = 1 / basis.dim * fft(u0)
                A_u0_til = A_k * u0_til
                u = basis.dim * ifft(A_u0_til)
                return u

        elif isinstance(basis, HermiteBasis):
            # Compute the Gauss-Hermite quadrature with n = dim + 1 points
            # for exact integration.

            A_operator_matrix = self.get_A_matrix(basis)

            def A_operator(u0):
                return A_operator_matrix @ u0

        return A_operator

    def get_B_operator(self, basis):
        x = basis.get_grid()
        # V_x = self.V(x) if self.V is not None else 0

        def B_operator(u0):
            return -1j * u0**2 *self.beta
        return B_operator


    def get_A_matrix(self, basis, spectral=False):

        if isinstance(basis, HermiteBasis):
            # Compute the Gauss-Hermite quadrature with n = dim + 1 points
            # for exact integration.
            n = basis.dim + 1
            k, w = hermite_nodes(n, weights=True)
            A_k = self.A_symbol(k / basis.scaling)
            phi_quad = hermite_matrix(n-1, k)
            A_matrix = (w * A_k * phi_quad) @ phi_quad.T
            for row in range(n-1):
                for col in range(n-1):
                    A_matrix[row, col] = (-1j)**(row-col) * A_matrix[row, col]

            if spectral:
                return A_matrix.T
            else:
                phi = hermite_matrix(n-1)
                c2h = phi * basis.weights
                h2c = phi.T

                return h2c @ A_matrix.T @ c2h
    
    def get_RHS(self, basis):
        A_func = self.get_A_operator(basis)
        B_func = self.get_B_operator(basis)
        
        def RHS(u0):
            RHS.nfev += 1
            return A_func(u0) + B_func(u0)
        
        RHS.nfev = 0
        
        return RHS
        

    def get_propagators(self, basis):
        """
        Return the partial propagators A / B for the model according to basis.

        Parameters
        ----------
        basis : Basis
            An instance of a Basis class.

        Returns
        -------
        (P_A, P_B) : tuple of functions (partial propagators in given basis).

        """
        # If given, evaluate the potential at grid points
        x = basis.get_grid()
        # V_x = self.V(x) if self.V is not None else 0

        if isinstance(basis, FourierBasis):
            # For Fourier basis, evaluate the symbol at discrete wavenumbers
            k = 2 * pi * fftfreq(basis.dim,
                                 basis.period / basis.dim)
            A_k = self.A_symbol(k)

            # Propagator for A in Fourier basis
            def P_A(u0, dt):
                propagator = exp(-1j * A_k * dt)
                u0_til = 1 / basis.dim * fft(u0)
                u0_til_evol = propagator * u0_til
                u = basis.dim * ifft(u0_til_evol)
                P_A.nfev += 1
                return u

        elif isinstance(basis, HermiteBasis):
            # Compute the Gauss-Hermite quadrature with n = dim + 1 points
            # for exact integration.
            # n = basis.dim + 1
            # kn, wn = hermite_nodes(n, weights=True)
            # A_kn = self.A_symbol(kn / basis.scaling)
            # phi_quad = hermite_matrix(n-1, kn)
            # A_matrix = (wn * A_kn * phi_quad) @ phi_quad.T
            # for row in range(n-1):
            #     for col in range(n-1):
            #         A_matrix[row, col] = (-1j)**(row-col) * A_matrix[row, col]

            # C = -1j * A_matrix.T
            phi = hermite_matrix(basis.dim)
            c2h = phi * basis.weights
            h2c = phi.T
            C = -1j * self.get_A_matrix(basis, spectral=True)
            # A_matrix_space = phi.T @ A_matrix.T @ phi * basis.weights
            dt_list = []
            prop_list = []

            def P_A(u0, dt):
                if dt not in dt_list:
                # if (dt not in P_A.dt):
                    # P_A.dt.append(dt)
                    dt_list.append(dt)
                    expCdt = expm(C * dt)
                    # P_A.propagator.append(h2c @ (expCdt @ c2h))
                    prop_list.append(h2c @ expCdt @ c2h)
                    # prop_list.append( expCdt)
                # i = P_A.dt.index(dt)
                i = dt_list.index(dt)
                # u = P_A.propagator[i] @ u0
                u = prop_list[i] @ u0
                P_A.nfev += 1
                return u

            #P_A.dt = []
            #P_A.propagator = []
        P_A.nfev = 0

        def P_B(u0, dt):
            u = u0 / (1 + u0 * dt * self.beta)
            P_B.nfev += 1
            return u    
        
        P_B.nfev = 0
        return P_A, P_B
 
