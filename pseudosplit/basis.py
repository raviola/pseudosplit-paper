#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basis module.

Implements pseudospectral bases.
"""

import numpy as np
from scipy.fft import fft, ifft, fftshift
import hermite


class Basis:
    """Basis class."""

    def __init__(self, name, dim, bounds):

        self.name = name
        self.dim = dim
        self.bounds = tuple(bounds)

        self.nodes, self.weights = self.get_nodes_weights()

    def get_nodes(self) -> np.ndarray:
        """
        Return nodes of quadrature associated with basis.

        Returns
        -------
        nodes: ndarray
            Array of nodes.

        """
        return self.nodes

    def get_weights(self) -> np.ndarray:
        """
        Return weights of quadrature associated with basis.

        Returns
        -------
        weights: ndarray
            Array of weights.

        """
        return self.weights

    def get_nodes_weights(self) -> tuple[np.ndarray]:
        """
        Return nodes and weights of quadrature associated with basis.

        Returns
        -------
        (nodes, weights): tuple of ndarray
            Tuple of nodes and weights.

        """
        raise NotImplementedError

    def get_grid(self) -> np.ndarray:
        """
        Return the collocation (problem) grid.

        Returns
        -------
        grid: np.ndarray

        """
        raise NotImplementedError

    def forward(self, values: np.ndarray) -> np.ndarray:
        """
        Forward discrete transform.

        Transform values at nodes into spectral coefficients.

        Parameters
        ----------
        values : np.ndarray
            Array of values at nodes.

        Returns
        -------
        coeffs: ndarray
            Array of spectral coefficients.

        """
        raise NotImplementedError

    def backward(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Inverse discrete transform.

        Transform spectral coefficients into values at nodes.

        Parameters
        ----------
        coeffs : np.ndarray
            Array of spectral coefficients.

        Returns
        -------
        values: ndarray
            Values at nodes.
        """
        raise NotImplementedError

    def integrate(self, values: np.ndarray) -> complex:
        """
        Integrate using the quadrature associated with basis.

        Parameters
        ----------
        values : np.ndarray
            Array of values to integrate.

        Returns
        -------
        complex
            Integral calculated from values using basis quadrature formula.

        """
        return self.weights @ values

    def diff(self, values):
        raise NotImplementedError

    def interpolate(self, values: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Interpolate values of a function.

        Parameters
        ----------
        values : np.ndarray
            Function to interpolate, given by its values at nodes.
        x : np.ndarray
            Points at which to interpolate the given function.

        Returns
        -------
        interp: np.ndarray
            Function values at the interpolation points.

        """
        raise NotImplementedError

    def __eq__(self, other):
        return self.dim == other.dim and self.bounds == other.bounds

class HermiteBasis(Basis):
    """Hermite functions basis."""

    name_prefix = 'hermite'
    def __init__(self, name, dim, bounds=(0., 1.)):
        # Hermite basis has default bounds
        self.center = bounds[0]
        self.scaling = bounds[1]

        super().__init__(name, dim, bounds)

        self.forward_matrix = None
        self.backward_matrix = None
        self.hermite_matrix = None

    def get_nodes_weights(self):
        # Note: we calculate the _modified_ weights (not Hermite weights)
        nodes, weights = hermite.hermite_nodes(self.dim, weights=True)
        return nodes, weights

    def get_grid(self):
        return self.scaling * self.nodes + self.center

    def forward(self, values):

        if self.forward_matrix is None:
            if self.hermite_matrix is None:
                self.hermite_matrix = hermite.hermite_matrix(self.dim,
                                                             self.nodes)
            self.forward_matrix = self.hermite_matrix * self.weights

        return self.forward_matrix @ values

    def backward(self, coeffs):

        if self.backward_matrix is None:
            if self.hermite_matrix is None:
                self.hermite_matrix = hermite.hermite_matrix(self.dim,
                                                             self.nodes)
            self.backward_matrix = self.hermite_matrix.T

        return self.backward_matrix @ coeffs

    def interpolate(self, values, x):

        coeffs = self.forward(values)
        interp_matrix = hermite.hermite_matrix(self.dim,
                                               1 / self.scaling *
                                               (x - self.center)).T
        return interp_matrix @ coeffs

    def integrate(self, values):
        return self.scaling * self.weights @ values


class FourierBasis(Basis):
    """Fourier trigonometric functions basis."""

    name_prefix = 'fourier'

    def __init__(self, name, dim, bounds=(0., 2 * np.pi)):

        if bounds[0] > bounds[1]:
            self.bounds = np.array(bounds[::-1])
        else:
            self.bounds = np.array(bounds)

        self.period = self.bounds[1] - self.bounds[0]

        super().__init__(name, dim, bounds)

    def get_nodes_weights(self):

        weights = self.period / self.dim * np.ones(self.dim,
                                                   dtype=np.double)
        nodes = np.arange(0., 2*np.pi, 2*np.pi / self.dim, dtype=np.double)
        return nodes, weights

    def get_grid(self):
        return self.period / (2 * np.pi) * self.nodes + self.bounds[0]

    def forward(self, values):

        return 1 / self.dim * fft(values)

    def backward(self, coeffs):

        return self.dim * ifft(coeffs)

    def interpolate(self, values, x):

        # Modal interpolation
        coeff = fftshift(self.forward(values))
        phi = np.zeros((len(x), self.dim), dtype=np.complex)

        xn = (x - self.bounds[0]) * 2 * np.pi / self.period

        for n in range(self.dim):
            phi[:, n] = np.exp(1.0j * (n - self.dim // 2) * xn)

        return phi @ coeff        