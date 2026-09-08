"""Incremental Potential Contact barrier functions.

``D`` and ``D_hat`` are squared distances. All derivatives are with respect
to ``D``.
"""

import numpy as np
import warp as wp

from scalar_types import scalar


# Defaults retained for compatibility with the original one-argument helpers.
d2hat = scalar(1.0e-4)
dhat = scalar(1.0e-2)
kappa = scalar(1.0e-3)

_ZERO = wp.constant(scalar(0.0))
_TWO = wp.constant(scalar(2.0))
_FOUR = wp.constant(scalar(4.0))
_EPS = wp.constant(scalar(1.0e-20))
_LARGE = wp.constant(scalar(1.0e20))


@wp.func
def ipc_barrier(D: scalar, D_hat: scalar, stiffness: scalar) -> scalar:
    """Return ``-stiffness * (D-D_hat)^2 * log(D/D_hat)`` when active."""
    if D >= D_hat:
        return _ZERO
    if D <= _EPS or D_hat <= _EPS:
        return _LARGE * stiffness
    diff = D - D_hat
    return -stiffness * diff * diff * wp.log(D / D_hat)


@wp.func
def ipc_barrier_derivative(D: scalar, D_hat: scalar, stiffness: scalar) -> scalar:
    """First derivative of :func:`ipc_barrier` with respect to ``D``."""
    if D >= D_hat:
        return _ZERO
    if D <= _EPS or D_hat <= _EPS:
        return -_LARGE * stiffness
    diff = D - D_hat
    return -stiffness * (_TWO * diff * wp.log(D / D_hat) + diff * diff / D)


@wp.func
def ipc_barrier_derivative2(D: scalar, D_hat: scalar, stiffness: scalar) -> scalar:
    """Second derivative of :func:`ipc_barrier` with respect to ``D``."""
    if D >= D_hat:
        return _ZERO
    if D <= _EPS or D_hat <= _EPS:
        return _LARGE * stiffness
    diff = D - D_hat
    return -stiffness * (
        _TWO * wp.log(D / D_hat)
        + _FOUR * diff / D
        - diff * diff / (D * D)
    )


# Compatibility names used by the earlier ipctkwp examples.
@wp.func
def barrier(D: scalar) -> scalar:
    return ipc_barrier(D, d2hat, kappa)


@wp.func
def barrier_derivative(D: scalar) -> scalar:
    return ipc_barrier_derivative(D, d2hat, kappa)


@wp.func
def barrier_derivative2(D: scalar) -> scalar:
    return ipc_barrier_derivative2(D, d2hat, kappa)


def ipc_barrier_np(D, D_hat, stiffness):
    if D >= D_hat:
        return 0.0
    if D <= 1.0e-20 or D_hat <= 1.0e-20:
        return 1.0e20 * stiffness
    diff = D - D_hat
    return -stiffness * diff * diff * np.log(D / D_hat)


def ipc_barrier_derivative_np(D, D_hat, stiffness):
    if D >= D_hat:
        return 0.0
    if D <= 1.0e-20 or D_hat <= 1.0e-20:
        return -1.0e20 * stiffness
    diff = D - D_hat
    return -stiffness * (2.0 * diff * np.log(D / D_hat) + diff * diff / D)


def ipc_barrier_derivative2_np(D, D_hat, stiffness):
    if D >= D_hat:
        return 0.0
    if D <= 1.0e-20 or D_hat <= 1.0e-20:
        return 1.0e20 * stiffness
    diff = D - D_hat
    return -stiffness * (
        2.0 * np.log(D / D_hat) + 4.0 * diff / D - diff * diff / (D * D)
    )


def barrier_np(D):
    return ipc_barrier_np(D, float(d2hat), float(kappa))


def barrier_derivative_np(D):
    return ipc_barrier_derivative_np(D, float(d2hat), float(kappa))


def barrier_derivative2_np(D):
    return ipc_barrier_derivative2_np(D, float(d2hat), float(kappa))
