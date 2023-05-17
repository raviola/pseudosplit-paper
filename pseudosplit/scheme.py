#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:25:24 2022.

@author: lisandro
"""

from affine import (affineS2_step,
                    affineS4_step,
                    affineS6_step,
                    affineS8_step)

from symplectic import (strang_step,
                        ruth_step,
                        neri_step,
                        yoshida_6_step)

from scipy.integrate import complex_ode, solve_ivp

class Scheme():
    """Time integration scheme."""

    is_splitting = None


    def step(self, u0, dt):
        """
        Advance one integration step of time `dt` from initial state `u0`.

        Parameters
        ----------
        u0 : State
            Initial state.
        dt : float
            Step size.

        Returns
        -------
        u : State
            State after evolution
        """
        raise NotImplementedError


class SplittingScheme(Scheme):
    is_splitting = True

    def __init__(self, splitting_step=None, P_A=None, P_B=None):
        self.splitting_step = splitting_step
        self.P_A = P_A
        self.P_B = P_B

    def set_propagators(self, P_A, P_B):
        self.P_A = P_A
        self.P_B = P_B
    
    def step(self, u0, dt):
        return self.splitting_step(u0, self.P_A, self.P_B, dt)

class Strang(SplittingScheme):
    name = 'strang'
    low_factor = 1/2
    high_factor = 3/4
    def __init__(self, P_A=None, P_B=None):
        super().__init__(splitting_step=strang_step, P_A=P_A, P_B=P_B)

class Ruth(SplittingScheme):
    name = 'ruth'
    low_factor = 1
    high_factor = 1
    def __init__(self, P_A=None, P_B=None):
        super().__init__(splitting_step=ruth_step, P_A=P_A, P_B=P_B)

class Neri(SplittingScheme):
    name = 'neri'
    low_factor = 3/4
    high_factor = 7/8
    def __init__(self, P_A=None, P_B=None):
        super().__init__(splitting_step=neri_step, P_A=P_A, P_B=P_B)

class Yoshida6(SplittingScheme):
    name = 'yoshida6'
    low_factor = 7/8
    high_factor = 15/16
    def __init__(self, P_A=None, P_B=None):
        super().__init__(splitting_step=yoshida_6_step, P_A=P_A, P_B=P_B)

class AffineSym2(SplittingScheme):
    name = 'affineS2'
    low_factor = 1
    high_factor = 1
    def __init__(self, P_A=None, P_B=None):
        super().__init__(splitting_step=affineS2_step, P_A=P_A, P_B=P_B)


class AffineSym4(SplittingScheme):
    name = 'affineS4'
    low_factor = 1
    high_factor = 1
    def __init__(self, P_A=None, P_B=None):
        super().__init__(splitting_step=affineS4_step, P_A=P_A, P_B=P_B)


class AffineSym6(SplittingScheme):
    name = 'affineS6'
    low_factor = 1
    high_factor = 1
    def __init__(self, P_A=None, P_B=None):
        super().__init__(splitting_step=affineS6_step, P_A=P_A, P_B=P_B)


class AffineSym8(SplittingScheme):
    name = 'affineS8'
    low_factor = 1
    high_factor = 1
    def __init__(self, P_A=None, P_B=None):
        super().__init__(splitting_step=affineS8_step, P_A=P_A, P_B=P_B)


class DOPRI853(Scheme):
    name = 'dopri853'
    is_splitting = False
    def __init__(self, RHS=None):
        self.RHS = RHS
        if RHS is not None:
            self.stepper = complex_ode(RHS)
            #self.stepper.set_integrator('dop853')
            
    def set_RHS(self, RHS):
        self.RHS = RHS
        self.stepper = complex_ode(RHS)
        
    def step(self, u0, dt):
        self.stepper.set_initial_value(u0)
        self.stepper.set_integrator('dop853',
                                    atol=1e-16,
                                    rtol=1e-14)
        return self.stepper.integrate(dt)

class DOPRI5(Scheme):
    name = 'dopri5'
    is_splitting = False
    def __init__(self, RHS=None):
        self.RHS = RHS
        if RHS is not None:
            self.stepper = complex_ode(RHS)
            #self.stepper.set_integrator('dopri5')
            
    def set_RHS(self, RHS):
        self.RHS = RHS
        self.stepper = complex_ode(RHS)
        
    def step(self, u0, dt):
        self.stepper.set_initial_value(u0)
        self.stepper.set_integrator('dopri5',
                                    atol=1e-10,
                                    rtol=1e-8)
        return self.stepper.integrate(dt)