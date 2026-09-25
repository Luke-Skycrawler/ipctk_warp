"""Warp edge-edge mollifier used by IPC for nearly parallel edges."""

import warp as wp

from scalar_types import vec4, mat12, scalar, vec3, vec12, vec6, mat6, make_vec6


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


@wp.func
def _cross_norm2_gradient_hessian_psd(edge_a: vec3, edge_b: vec3):
    """Gradient and analytical PSD Hessian of ||edge_a x edge_b||^2."""
    zero = scalar(0.0)
    one = scalar(1.0)
    aa_sq = wp.dot(edge_a, edge_a)
    bb_full_sq = wp.dot(edge_b, edge_b)
    eps = scalar(1.0e-30)

    if aa_sq <= eps:
        hess = mat6(zero)
        projector = _TWO * (bb_full_sq * wp.diag(vec3(one)) - wp.outer(edge_b, edge_b))
        for i in range(3):
            for j in range(3):
                hess[i, j] = projector[i, j]
        return vec6(zero), hess

    alpha = wp.dot(edge_a, edge_b) / aa_sq
    b = edge_b - alpha * edge_a
    bb_sq = wp.dot(b, b)

    if bb_sq <= eps:
        hess = mat6(zero)
        projector = _TWO * (aa_sq * wp.diag(vec3(one)) - wp.outer(edge_a, edge_a))
        for i in range(3):
            for j in range(3):
                hess[i, j] = alpha * alpha * projector[i, j]
                hess[i, j + 3] = -alpha * projector[i, j]
                hess[i + 3, j] = -alpha * projector[i, j]
                hess[i + 3, j + 3] = projector[i, j]
        return vec6(zero), hess

    n = wp.normalize(wp.cross(edge_a, b))
    aa = wp.sqrt(aa_sq)
    bb = wp.sqrt(bb_sq)
    eigenvalues = vec6(zero)
    eigenvalues[0] = bb_sq
    eigenvalues[1] = aa_sq
    eigenvalues[2] = aa * bb
    eigenvalues[3] = -aa * bb
    term = aa_sq + bb_sq
    discriminant = term * term + scalar(12.0) * aa_sq * bb_sq
    eigenvalues[4] = scalar(0.5) * (term + wp.sqrt(discriminant))
    eigenvalues[5] = scalar(0.5) * (term - wp.sqrt(discriminant))

    eigenvectors = mat6(zero)
    zero3 = vec3(zero)
    eigenvectors[0] = make_vec6(n, zero3)
    eigenvectors[1] = make_vec6(zero3, n)
    eigenvectors[2] = make_vec6(-b / bb, edge_a / aa)
    eigenvectors[3] = make_vec6(b / bb, edge_a / aa)
    for i in range(4, 6):
        eigenvectors[i] = make_vec6(
            _TWO * bb * edge_a,
            (eigenvalues[i] / bb - bb) * b,
        )
    for i in range(6):
        eigenvectors[i] /= wp.length(eigenvectors[i])
        eigenvalues[i] = _TWO * wp.max(eigenvalues[i], zero)

    grad_a = scalar(-2.0) * wp.cross(b, wp.cross(b, edge_a))
    grad_b = scalar(-2.0) * wp.cross(edge_a, wp.cross(edge_a, b))
    grad = make_vec6(grad_a - alpha * grad_b, grad_b)

    eigenvectors = wp.transpose(eigenvectors)
    transform = mat6(
        one, zero, zero, zero, zero, zero,
        zero, one, zero, zero, zero, zero,
        zero, zero, one, zero, zero, zero,
        -alpha, zero, zero, one, zero, zero,
        zero, -alpha, zero, zero, one, zero,
        zero, zero, -alpha, zero, zero, one,
    )
    eigenvectors = wp.transpose(transform) @ eigenvectors
    return grad, eigenvectors @ wp.diag(eigenvalues) @ wp.transpose(eigenvectors)


@wp.func
def ee_mollifier_gradient_hessian_psd(
    x0: vec3, x1: vec3, x2: vec3, x3: vec3, eps_x: scalar
):
    """Exact mollifier gradient and an analytical PSD Hessian approximation."""
    edge_a = x1 - x0
    edge_b = x3 - x2
    grad_edges, hess_edges = _cross_norm2_gradient_hessian_psd(edge_a, edge_b)

    grad_x = vec12()
    hess_x = mat12()
    for i in range(12):
        edge_i = i
        sign_i = _ONE
        if i < 3:
            sign_i = -_ONE
        elif i < 6:
            edge_i = i - 3
        elif i < 9:
            edge_i = i - 3
            sign_i = -_ONE
        else:
            edge_i = i - 6
        grad_x[i] = sign_i * grad_edges[edge_i]

        for j in range(12):
            edge_j = j
            sign_j = _ONE
            if j < 3:
                sign_j = -_ONE
            elif j < 6:
                edge_j = j - 3
            elif j < 9:
                edge_j = j - 3
                sign_j = -_ONE
            else:
                edge_j = j - 6
            hess_x[i, j] = sign_i * sign_j * hess_edges[edge_i, edge_j]

    grad = vec12()
    hess = mat12()
    cross = wp.cross(edge_a, edge_b)
    x = wp.dot(cross, cross)
    if eps_x > _ZERO and x < eps_x:
        inv_eps = _ONE / eps_x
        derivative = _TWO * inv_eps * (_ONE - x * inv_eps)
        for i in range(12):
            grad[i] = derivative * grad_x[i]
            for j in range(12):
                # The omitted scalar second-derivative term is NSD because
                # the active mollifier polynomial is concave in x.
                hess[i, j] = derivative * hess_x[i, j]
    return grad, hess


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
