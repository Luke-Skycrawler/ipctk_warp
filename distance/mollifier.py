"""Warp edge-edge mollifier used by IPC for nearly parallel edges."""

import warp as wp

from scalar_types import vec4, mat12, scalar, vec3, vec12


_ZERO = wp.constant(scalar(0.0))
_ONE = wp.constant(scalar(1.0))
_TWO = wp.constant(scalar(2.0))
_MOLLIFIER_COEFF = wp.constant(scalar(1.0e-3))


@wp.func
def _ee_cross_norm2(x0: vec3, x1: vec3, x2: vec3, x3: vec3) -> scalar:
    a = x1 - x0
    b = x3 - x2
    cross = wp.cross(a, b)
    return wp.dot(cross, cross)


@wp.func
def _ee_cross_norm2_derivatives(x0: vec3, x1: vec3, x2: vec3, x3: vec3):
    """Return x, gradient and Hessian w.r.t. [x0, x1, x2, x3]."""
    a = x1 - x0
    b = x3 - x2
    aa = wp.dot(a, a)
    ab = wp.dot(a, b)
    bb = wp.dot(b, b)
    cross = wp.cross(a, b)
    value = wp.dot(cross, cross)

    grad_a = _TWO * (bb * a - ab * b)
    grad_b = _TWO * (aa * b - ab * a)
    grad = vec12()
    for k in range(3):
        grad[k] = -grad_a[k]
        grad[3 + k] = grad_a[k]
        grad[6 + k] = -grad_b[k]
        grad[9 + k] = grad_b[k]

    # Hessian blocks for a and b, mapped to the four endpoints by signs.
    hess = mat12()
    signs_a = vec4(scalar(-1.0), _ONE, _ZERO, _ZERO)
    signs_b = vec4(_ZERO, _ZERO, scalar(-1.0), _ONE)
    for vi in range(4):
        for vj in range(4):
            for i in range(3):
                for j in range(3):
                    identity = _ZERO
                    if i == j:
                        identity = _ONE
                    Haa = _TWO * (bb * identity - b[i] * b[j])
                    Hbb = _TWO * (aa * identity - a[i] * a[j])
                    Hab = _TWO * (
                        _TWO * a[i] * b[j] - b[i] * a[j] - ab * identity
                    )
                    Hba = _TWO * (
                        _TWO * b[i] * a[j] - a[i] * b[j] - ab * identity
                    )
                    hess[3 * vi + i, 3 * vj + j] = (
                        signs_a[vi] * signs_a[vj] * Haa
                        + signs_a[vi] * signs_b[vj] * Hab
                        + signs_b[vi] * signs_a[vj] * Hba
                        + signs_b[vi] * signs_b[vj] * Hbb
                    )
    return value, grad, hess


@wp.func
def ee_mollifier_threshold(
    x0_rest: vec3, x1_rest: vec3, x2_rest: vec3, x3_rest: vec3
) -> scalar:
    """Compute ``1e-3 ||a_rest||^2 ||b_rest||^2``."""
    a = x1_rest - x0_rest
    b = x3_rest - x2_rest
    return _MOLLIFIER_COEFF * wp.dot(a, a) * wp.dot(b, b)


@wp.func
def ee_mollifier_value(
    x0: vec3, x1: vec3, x2: vec3, x3: vec3, eps_x: scalar
) -> scalar:
    x = _ee_cross_norm2(x0, x1, x2, x3)
    if eps_x > _ZERO and x < eps_x:
        r = x / eps_x
        return (_TWO - r) * r
    return _ONE


@wp.func
def ee_mollifier_derivatives(
    x0: vec3, x1: vec3, x2: vec3, x3: vec3, eps_x: scalar
):
    """Return mollifier value, gradient and exact Hessian."""
    value = _ONE
    grad = vec12()
    hess = mat12()
    x = _ee_cross_norm2(x0, x1, x2, x3)
    if eps_x > _ZERO and x < eps_x:
        x, grad_x, hess_x = _ee_cross_norm2_derivatives(x0, x1, x2, x3)
        inv_eps = _ONE / eps_x
        r = x * inv_eps
        value = (_TWO - r) * r
        derivative = _TWO * inv_eps * (_ONE - r)
        derivative2 = -_TWO * inv_eps * inv_eps
        for i in range(12):
            grad[i] = derivative * grad_x[i]
            for j in range(12):
                hess[i, j] = (
                    derivative2 * grad_x[i] * grad_x[j]
                    + derivative * hess_x[i, j]
                )
    return value, grad, hess


# Aliases matching warp-ipc's public kernel names.
@wp.func
def ee_mollifier(
    x0: vec3, x1: vec3, x2: vec3, x3: vec3, eps_x: scalar
) -> scalar:
    return ee_mollifier_value(x0, x1, x2, x3, eps_x)


@wp.func
def ee_mollifier_gradient(
    x0: vec3, x1: vec3, x2: vec3, x3: vec3, eps_x: scalar
) -> vec12:
    value, grad, hess = ee_mollifier_derivatives(x0, x1, x2, x3, eps_x)
    return grad


@wp.func
def ee_mollifier_hessian(
    x0: vec3, x1: vec3, x2: vec3, x3: vec3, eps_x: scalar
) -> mat12:
    value, grad, hess = ee_mollifier_derivatives(x0, x1, x2, x3, eps_x)
    return hess
