"""Hermite functions module."""

import numpy as np
from numpy import pi, sqrt, exp, log, abs
from scipy.special import roots_hermite


def hermite_function(n, x):
    """Return the (normalized) `n`-th Hermite function evaluated at `x`.

    Parameters
    ----------
    n : integer
        Degree of Hermite function.

    x : ndarray
        Array of values.

    Returns
    -------
    ndarray
        The n-th (normalized) Hermite function evaluated at x.

    Notes
    -----
    The function uses a modified three-term recursion formula to avoid
    overflow / underflow issues that appear when doing a naive evaluation
    of high-order Hermite functions. See [1]_.

    References
    ----------
    .. [1] B. Bunck, "A fast algorithm for evaluation of normalized Hermite
        functions", BIT Numer. Math. (2009) 49: 281–295
    """
    if n == 0:
        return np.ones_like(x) * pi**(-0.25) * exp(-x**2 / 2)
    if n == 1:
        return sqrt(2.) * x * pi**(-0.25) * exp(-x**2 / 2)

    h_i_2 = np.ones_like(x) * pi**(-0.25)
    h_i_1 = sqrt(2.) * x * pi**(-0.25)

    sum_log_scale = np.zeros_like(x)

    for i in range(2, n + 1):

        h_i = sqrt(2. / i) * x * h_i_1 - sqrt((i - 1.) / i) * h_i_2
        h_i_2, h_i_1 = h_i_1, h_i

        abs_h_i=np.abs(h_i)
        # Avoid calculating log of zero entries
        log_scale = np.log(abs_h_i,out=np.zeros_like(abs_h_i),where=abs_h_i>1).round()
        scale = exp(-log_scale)

        h_i = h_i * scale
        h_i_1 = h_i_1 * scale
        h_i_2 = h_i_2 * scale

        sum_log_scale += log_scale
    return h_i * exp(-x**2 / 2 + sum_log_scale)


def hermite_nodes(n, weights=False):
    """Return the zeros and weights of the Hermite function of order `n`.

    Uses the Golub-Welsh algorithm.

    TODO: documentation
    """
    # x = eigvalsh_tridiagonal(np.zeros(n), sqrt(np.arange(1, n) / 2.))
    x, _ = roots_hermite(n)
    if weights is True:
        h_n_1_nodes = hermite_function(n - 1, x)
        w = 1 / n / h_n_1_nodes ** 2
        return x, w
    return x


def hermite_modified_weights(n):
    """Calculate modified weights of the Hermite-Gauss quadrature.

    TODO: documentation
    """
    x, _ = roots_hermite(n)
    h_n_1_nodes = hermite_function(n - 1, x)
    w = 1 / n / h_n_1_nodes ** 2

    return w


def hermite_matrix(n, x=None):
    """Return the matrix of Hermite functions evaluated at the given points.

    TODO: documentation
    """
    # If the collocation points aren't specified, default to quadrature nodes
    if x is None:
        x, _ = roots_hermite(n)
        phi = np.empty((n, n), dtype=np.double)
    else:
        phi = np.empty((n, x.shape[0]), dtype=np.double)

    phi[0, :] = np.ones_like(x) * pi**(-0.25) * exp(-x**2 / 2)
    phi[1, :] = sqrt(2.) * x * pi**(-0.25) * exp(-x**2 / 2)

    h_i_2 = np.ones_like(x) * pi**(-0.25)
    h_i_1 = sqrt(2.) * x * pi**(-0.25)
    sum_log_scale = np.zeros_like(x)

    for i in range(2, n):

        h_i = sqrt(2./i) * x * h_i_1 - sqrt((i - 1.) / i) * h_i_2

        h_i_2, h_i_1 = h_i_1, h_i
        abs_h_i=np.abs(h_i)
        # Avoid calculating log of zero entries
        log_scale = np.log(abs_h_i,out=np.zeros_like(abs_h_i),where=abs_h_i>1).round()
        scale = exp(-log_scale)
        h_i = h_i * scale
        h_i_1 = h_i_1 * scale
        h_i_2 = h_i_2 * scale
        sum_log_scale += log_scale

        phi[i, :] = h_i * exp(-x**2 / 2 + sum_log_scale)

    return phi
