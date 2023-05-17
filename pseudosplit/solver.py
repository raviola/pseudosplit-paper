"""
Created on Wed Oct 19 17:49:01 2022.

@author: lisandro
"""

from state import State

class Solver():
    """Class for solving 1D evolution equations.

    The solver uses a split-step scheme on a problem discretized by means of a
    pseudospectral (or other discrete) representation in space.
    """

    def __init__(self, model, scheme):

        self.model = model
        self.scheme = scheme
        self.state = None
        self.active = False

    def start(self, init_state, init_time, end_time):

        self.state = init_state
        self.sim_time = init_time
        self.end_time = end_time

        self._basis = init_state._basis

        if self.scheme.is_splitting:
            self.P_A, self.P_B = self.model.get_propagators(self._basis)
            self.scheme.set_propagators(self.P_B, self.P_A)
        else:
            RHS = self.model.get_RHS(self._basis)
            def f(t, y):
                return -1j * RHS(y)
            wrapped_RHS = f
            self.scheme.set_RHS(wrapped_RHS)

        self.active = True

    def step(self, dt):
        if self.state is not None:
            if self.active:
                if self.sim_time >= self.end_time:
                    self.active = False
                    return self.state
                elif self.sim_time + dt >= self.end_time:
                    dt = self.end_time - self.sim_time
                    self.active = False
                u0 = self.state.values
                u = self.scheme.step(u0, dt)
                self.state = State('u', self.state._basis)
                self.state.values = u
                self.sim_time += dt
                return self.state
        else:
            raise ValueError("The state hasn't been specified")

    def solve(init_state, time_points):
        pass
