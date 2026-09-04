import numpy as np 
import ipctk 
import warp as wp
from scalar_types import *
wp.config.max_unroll = 1
wp.config.enable_backward = False

psd_project = -1
@wp.func 
def point_edge_distance(p: vec3,  edge0: vec3, edge1: vec3): 

    e0 = edge1 - edge0
    e2 = p - edge0


    alpha = wp.dot(e2, e0) / wp.dot(e0, e0)

    q = edge0 + alpha * e0
    d = wp.length(p - q)    
    return alpha, d 

@wp.func 
def point_edge_distance_gradient_hessian(p: vec3, edge0: vec3, edge1: vec3):
    z = scalar(0.0)
    o = scalar(1.0)
    alpha, d = point_edge_distance(p, edge0, edge1)
    e0 = edge1 - edge0
    e2 = p - edge0
    
    e2p = e2 - alpha * e0 
    e1p = wp.normalize(wp.cross(e0, e2p))
    
    lambdas = vec3(o) * scalar(2.0)
    
    e2p_unit = wp.normalize(e2p)
    e0_unit = wp.normalize(e0)
    
    # last 3 rows of q. The first 6 rows are zero
    qs3 = wp.matrix_from_cols(e2p_unit, e1p, e0_unit)
    
    
    # ker_pcpx_simple = mat33(
    #     z, -o, o,
    #     z, z, z,
    #     o, alpha - o, -alpha
    # )


    dadx = dalphadx(e0, e2)

    k2 = vec3(o, alpha - o, -alpha)
    grad = wp.outer(k2, e2p_unit) * scalar(2.0) * d
    # grad = wp.matrix_from_rows(e2p_unit, (alpha - o) * e2p_unit, -alpha * e2p_unit) * 2.0 * d
    # flatten it by rows to get the 9x1 gradient 

    k2k2T = wp.outer(k2, k2)
    Hu33 = qs3 @ wp.diag(lambdas) @ wp.transpose(qs3)
    
    e0THue0 = wp.dot(Hu33 @ e0, e0)
    # hess_delta = outer(dadx, dadx) * e0^T H e0
    # out9x9 = wp.outer(dadx, dadx) * e0THue0    
    out9x9 = wp.matrix(
        z, 
        shape = (9, 9),
        dtype = scalar
    )

    # The exact Hessian is S - N, where
    #   S = 2 * (k2 k2^T) \otimes I_3
    #   N = e0THue0 * vec(dadx) vec(dadx)^T.
    # S and N are not, in general, simultaneously diagonalizable.  Their
    # combined eigensystem consists of two unchanged positive modes of S,
    # five zero modes, and a 2x2 system coupling the projections of dadx onto
    # range(S) and null(S).
    k2_norm_sq = wp.dot(k2, k2)
    s = scalar(2.0) * k2_norm_sq
    identity = wp.diag(vec3(o))

    # P is the projector onto range(S).  Apply it to vec(dadx) without
    # explicitly constructing a 9x9 matrix.
    y = k2[0] * dadx[0] + k2[1] * dadx[1] + k2[2] * dadx[2]
    a = wp.matrix_from_rows(
        k2[0] * y / k2_norm_sq,
        k2[1] * y / k2_norm_sq,
        k2[2] * y / k2_norm_sq,
    )
    b = dadx - a
    a_norm_sq = wp.dot(a[0], a[0]) + wp.dot(a[1], a[1]) + wp.dot(a[2], a[2])
    b_norm_sq = wp.dot(b[0], b[0]) + wp.dot(b[1], b[1]) + wp.dot(b[2], b[2])

    eps = scalar(1.0e-30)
    a_hat = a
    b_hat = b
    q_plus = a
    q_minus = b
    lambda_plus = scalar(0.0)
    lambda_minus = scalar(0.0)

    if a_norm_sq > eps:
        a_hat = a / wp.sqrt(a_norm_sq)
    if b_norm_sq > eps:
        b_hat = b / wp.sqrt(b_norm_sq)

    if a_norm_sq > eps and b_norm_sq > eps:
        # Matrix of H in the orthonormal basis (a_hat, b_hat).
        h00 = s - e0THue0 * a_norm_sq
        h01 = -e0THue0 * wp.sqrt(a_norm_sq * b_norm_sq)
        h11 = -e0THue0 * b_norm_sq

        theta = scalar(0.5) * wp.atan2(scalar(2.0) * h01, h00 - h11)
        cos_theta = wp.cos(theta)
        sin_theta = wp.sin(theta)
        q_plus = cos_theta * a_hat + sin_theta * b_hat
        q_minus = -sin_theta * a_hat + cos_theta * b_hat

        discriminant = wp.sqrt(
            (h00 - h11) * (h00 - h11) + scalar(4.0) * h01 * h01
        )
        lambda_plus = scalar(0.5) * (h00 + h11 + discriminant)
        lambda_minus = scalar(0.5) * (h00 + h11 - discriminant)
    elif a_norm_sq > eps:
        # N lies entirely in range(S), so only the a_hat mode changes.
        q_plus = a_hat
        q_minus = a_hat
        lambda_plus = s - e0THue0 * a_norm_sq
        lambda_minus = lambda_plus
    elif b_norm_sq > eps:
        # S and N have orthogonal ranges in this limiting case.
        q_plus = b_hat
        q_minus = b_hat
        lambda_plus = -e0THue0 * b_norm_sq
        lambda_minus = lambda_plus

    for ii in range(3):
        for jj in range(3):
            if wp.static(psd_project == -1):
                block = wp.outer(q_minus[ii], q_minus[jj]) * wp.min(lambda_minus, z)
                if b_norm_sq <= eps and a_norm_sq > eps:
                    block = wp.outer(a_hat[ii], a_hat[jj]) * wp.min(lambda_minus, z)
            elif wp.static(psd_project == 1):
                # Two directions in range(S), orthogonal to a, retain the
                # eigenvalue s.  The remaining positive mode is q_plus.
                block = s * (
                    k2k2T[ii, jj] * identity / k2_norm_sq
                    - wp.outer(a_hat[ii], a_hat[jj])
                )
                block += wp.outer(q_plus[ii], q_plus[jj]) * wp.max(lambda_plus, z)

                if a_norm_sq <= eps:
                    # There is no direction to remove from range(S).
                    block = s * k2k2T[ii, jj] * identity / k2_norm_sq
            else:
                block = -wp.outer(dadx[ii], dadx[jj]) * e0THue0 + k2k2T[ii, jj] * Hu33 

            for kk in range(3):
                for ll in range(3):
                    out9x9[ii * 3 + kk, jj * 3 + ll] = block[kk, ll]
    
    return grad, out9x9
    
@wp.func 
def dalphadx(e0: vec3, e2: vec3):
    term = scalar(1.0) / wp.dot(e0, e0)
    term2 = scalar(2.0) * e0 * wp.dot(e0, e2) / wp.dot(e0, e0)
    
    return wp.matrix_from_rows(term * e0, term * (term2 - e2 - e0), term * (e2 - term2))
    
@wp.kernel
def point_edge_distance_gradient_hessian_kernel(x: wp.array2d(dtype = vec3), out_grad: wp.array(dtype = mat33), out_hess: wp.array(dtype = mat99)):
    i = wp.tid()
    p = x[i, 0]
    edge0 = x[i, 1]
    edge1 = x[i, 2]

    grad, hess = point_edge_distance_gradient_hessian(p, edge0, edge1)
    out_grad[i] = grad
    out_hess[i] = hess

def test_warp(grad, hess, p, edge0, edge1, verbose = True):
    ipc_ref = ipctk.point_line_distance_hessian(p, edge0, edge1)

    # diff = ipc_ref - hess

    ipc_grad = ipctk.point_line_distance_gradient(p, edge0, edge1)
    grad_diff = ipc_grad - grad

    eigvals, eigvecs = np.linalg.eigh(ipc_ref)
    if psd_project == 1:
        eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
        hess_ref = eigvecs @ np.diag(eigvals) @ eigvecs.T
    elif psd_project == -1:
        eigvals = np.clip(eigvals, a_min=None, a_max=0.0)
        hess_ref = eigvecs @ np.diag(eigvals) @ eigvecs.T
    else:
        hess_ref = ipc_ref

    diff = hess_ref - hess

    if verbose:
        print(f"ipc ref norm = {np.linalg.norm(ipc_ref)}, hess norm = {np.linalg.norm(hess)}, diff norm = {np.linalg.norm(diff)}")
        print(f"ipc grad norm = {np.linalg.norm(ipc_grad)}, grad norm = {np.linalg.norm(grad)}, diff norm = {np.linalg.norm(grad_diff)}")
    return grad_diff, diff

def test(p, edge0, edge1):
    
    e0 = edge1 - edge0
    e2 = p - edge0


    alpha = np.dot(e2, e0) / np.dot(e0, e0)
    t = np.clip(alpha, 0.0, 1.0)

    q = edge0 + t * e0
    d = np.linalg.norm(p - q)



    e2p = e2 - alpha * e0
    e1p = np.cross(e0, e2p)

    e1p /= np.linalg.norm(e1p)
    
    lambdas = np.zeros((3,))
    lambdas[0] = 2
    lambdas[1] = 2
    lambdas[2] = 2

    e2p_unit = e2p / np.linalg.norm(e2p)
    e0_unit = e0 / np.linalg.norm(e0)
    
    
    qs = np.zeros((9, 3))
    z6 = np.zeros((6,))

    qs[:, 0] = np.concatenate([z6, e2p_unit])
    qs[:, 1] = np.concatenate([z6, e1p])
    qs[:, 2] = np.concatenate([z6, e0])

    for i in range(3):
        qs[:, i] /= np.linalg.norm(qs[:, i])

    ker_pcpx_simple = np.array([
        [0, 0, 1],
        [-1, 0, alpha - 1],
        [1, 0, -alpha]
    ]).T
    pcpx_simple = np.kron(ker_pcpx_simple, np.eye(3))

    gu = qs[:, 0]
    grad = pcpx_simple.T @ gu * 2.0 * d
    ref_grad = ipctk.point_line_distance_gradient(p, edge0, edge1)
    # print(f"grad = {grad}, ref_grad = {ref_grad}")
    print(f"grad diff norm = {np.linalg.norm(grad - ref_grad)}")
    
    pcpx_delta = np.zeros((9, 9))

    term = 1 / np.dot(e0, e0)
    e0de2 = np.dot(e0_unit, e2)
    term2 = 2 * e0_unit * e0de2
    dalphadx = np.concatenate([
        e0,
        term2 - e2 - e0,
        e2 - term2
    ]) * term

    Hu = qs @ np.diag(lambdas) @ qs.T
    Hu33 = Hu[6:9, 6:9]
    pcpx_delta[6:9, :] = np.outer(e0, dalphadx)
    left = np.outer(ker_pcpx_simple[2], ker_pcpx_simple[2])
    simple_hess = np.kron(left, Hu33)

    e0THue0 = e0 @ Hu33 @ e0
    delta_hess = np.outer(dalphadx, dalphadx) * e0THue0
    # hess = pcpx_simple.T @ Hu @ pcpx_simple - pcpx_delta.T @ Hu @ pcpx_delta
    hess = simple_hess - delta_hess

    A2 = delta_hess
    print(f"A2 norm = {np.linalg.norm(A2)}")
    
    ipc_ref = ipctk.point_line_distance_hessian(p, edge0, edge1)

    diff = ipc_ref - hess 
    print(f"ipc ref norm = {np.linalg.norm(ipc_ref)}, hess norm = {np.linalg.norm(hess)}, diff norm = {np.linalg.norm(diff)}")

    # print(f"ipc ref = \n{ipc_ref}\n\nhess = \n{hess}\ndiff = \n{diff}")

if __name__ == "__main__":
    # x = np.load("pe.npz")

    # p = np.array([0, 1, 0], dtype=float)
    # edge0 = np.array([-1, 0, 0], dtype=float)
    # edge1 = np.array([1, 0, 0], dtype=float)

    data = np.load("pe.npz")
    n_tests = len(data["x"])

    xnp = data["x"][:n_tests]
    
    x = wp.zeros((n_tests, 3), dtype = vec3)
    x.assign(xnp)
    grad = wp.zeros((n_tests, ), dtype = mat33)
    hess = wp.zeros((n_tests, ), dtype = mat99)

    wp.launch(point_edge_distance_gradient_hessian_kernel, dim = (n_tests, ), inputs = [x, grad, hess])    
    

    gradnp = grad.numpy()
    hessnp = hess.numpy()
    verbose = n_tests < 50

    diff_h = []
    diff_g = []
    for i in range(n_tests):
        p = xnp[i, 0]
        edge0 = xnp[i, 1]
        edge1 = xnp[i, 2]

        # test(p, edge0, edge1)
        dg, dh = test_warp(gradnp[i].reshape(-1), hessnp[i], p, edge0, edge1, verbose=verbose)
        diff_h.append(np.linalg.norm(dh))
        diff_g.append(np.linalg.norm(dg))

    diff_h = np.array(diff_h)
    diff_g = np.array(diff_g)
    print(f"Average, max diff hess norm: {np.mean(diff_h)}, {np.max(diff_h)}")
    print(f"Average, max diff grad norm: {np.mean(diff_g)}, {np.max(diff_g)}")
    assert np.max(diff_h) < 1.0e-10
    assert np.max(diff_g) < 1.0e-10
