"""State module."""

from typing import Callable
import numpy as np
from basis import Basis


class State():
    """Class for states."""

    def __init__(self, name: str, basis: Basis, u: Callable = None):

        self.name = name
        self._basis = basis
        self._grid = basis.get_grid()
        self._values = None if u is None else u(self._grid)
        self._coeffs = None

    @property
    def basis(self):
        """
        Return the grid (collocation points).

        Returns
        -------
        grid: np.ndarray
            Array of collocation points
        """
        return self._grid[:]

    @property
    def grid(self):
        """
        Return the grid (collocation points).

        Returns
        -------
        grid: np.ndarray
            Array of collocation points
        """
        return self._grid[:]

    @property
    def values(self):
        """
        Return state values on the grid.

        Returns
        -------
        values: np.ndarray
            Array of values of state at the collocation grid
        """
        return self._values[:] if self._values is not None else None

    @values.setter
    def values(self, values):
        self._values = values[:]

    @property
    def coeffs(self):
        """
        Return the spectral coefficients for the state.

        Returns
        -------
        coeffs: np.ndarray
            Spectral coefficients

        """
        coeffs = self._basis.forward(self._values)
        self._coeffs = coeffs
        return coeffs[:]

    def norm(self, p: int = 2) -> float:
        """
        Return the p-norm of state.

        Parameters
        ----------
        p : int, optional
            Order of norm. The default is 2.

        Returns
        -------
        norm : float
            Non-negative value of norm.

        """
        norm = self._basis.integrate(np.abs(self._values)**p)**(1 / p)
        return norm

    def dot(self, other) -> complex:
        """
        Return the (complex) dot product with `other`.

        Parameters
        ----------
        other : State
            The second element for the dot product (the first is `self`).

        Returns
        -------
        dot : complex
            (Complex) dot product.

        """
        dot = self._basis.integrate(self._values * np.conj(other.values))
        return dot

    def integrate(self):
        return self._basis.integrate(self._values)

    def interpolate(self, x: np.ndarray) -> np.ndarray:
        return self._basis.interpolate(self._values, x)

    def diff(self):
        raise NotImplementedError