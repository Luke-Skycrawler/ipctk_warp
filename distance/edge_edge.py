import warp as wp
from scalar_types import * 
from .hl import signed_distance, eig_Hl, gl
from .ee import C_ee, dceedx_s
import numpy as np

wp.config.max_unroll = 1
wp.config.enable_backward = False

psd_project = -1

@wp.func
def x_to_grad_psd_hess_ee(x0: vec3, x1: vec3, x2: vec3, x3: vec3): 
    e0p, e1p, e2p = C_ee(x0, x1, x2, x3)

    lams = wp.vector(length=5, dtype = scalar)
    qs = wp.zeros((5, 3), dtype = vec3)
    lam0, lam1, lam2, lam3 = eig_Hl(e0p, e1p, e2p, qs)
    l = signed_distance(e0p, e1p, e2p)

    # clamp to positive
    if wp.static(psd_project == 1):
        lams[0] = wp.max(lam0, scalar(0.0))
        lams[1] = wp.max(lam1, scalar(0.0))
        lams[2] = wp.max(lam2, scalar(0.0))
        lams[3] = wp.max(lam3, scalar(0.0))
        lams[4] = scalar(2.0)
    elif wp.static(psd_project) == -1: 
        lams[0] = wp.min(lam0, scalar(0.0))
        lams[1] = wp.min(lam1, scalar(0.0))
        lams[2] = wp.min(lam2, scalar(0.0))
        lams[3] = wp.min(lam3, scalar(0.0))
        lams[4] = scalar(0.0)
    else: 
        lams[0] = lam0
        lams[1] = lam1
        lams[2] = lam2
        lams[3] = lam3
        lams[4] = scalar(2.0)

    gl0, gl1, gl2 = gl(l, e2p)
    qs[4, 0] = gl0
    qs[4, 1] = gl1
    qs[4, 2] = gl2

    for ii in range(5):         
        sum = scalar(0.0)
        for jj in range(3):
            sum += wp.length_sq(qs[ii, jj])
        sum = wp.sqrt(sum)
        for jj in range(3):
            qs[ii, jj] /= sum

    dcdxi = dceedx_s(x0, x1, x2, x3)
    u = wp.zeros((5, 4), dtype = vec3)

    
    
    out = mat12()
    for ii in range(5): 
        for kk in range(3): 
            for jj in range(4):
                u[ii, jj] += qs[ii, kk] * dcdxi[kk, jj]

    # grad = dcdx.T @ (gl0, gl1, gl2) * 2 * l
    grad = vec12()
    for jj in range(4):
        for kk in range(3):
            grad[jj * 3 + kk] = u[4, jj][kk] * scalar(2.0) * l
    
    # u^T @ L @ u
    tmp = wp.zeros((4, 4), dtype = mat33)
    for ii in range(5): 
        for jj in range(4):
            for kk in range(4):
                tmp[jj, kk] += lams[ii] * wp.outer(u[ii, jj], u[ii, kk])

    for ii in range(4):
        for jj in range(4):
            for mm in range(3):
                for nn in range(3):
                    out[ii * 3 + mm, jj * 3 + nn] = tmp[ii, jj][mm][nn]
    return grad, out

@wp.kernel
def _test(x: wp.array2d(dtype = vec3), out_grad: wp.array(dtype = vec12), out_hess: wp.array(dtype = mat12)):
    i = wp.tid()
    x0 = x[i, 0]
    x1 = x[i, 1]
    x2 = x[i, 2]
    x3 = x[i, 3]

    grad, hess = x_to_grad_psd_hess_ee(x0, x1, x2, x3)
    out_grad[i] = grad
    out_hess[i] = hess

if __name__ == "__main__":
    xnp = np.load("ee.npz")["x"]
    nee = xnp.shape[0]
    x = wp.zeros((nee, 4), dtype = vec3)
    x.assign(xnp)
    
    grad = wp.zeros((nee, ), dtype = vec12)
    hess = wp.zeros((nee, ), dtype = mat12)
    wp.launch(_test, dim = (nee, ), inputs = [x, grad, hess])
    import ipctk
    gnp = grad.numpy()
    hnp = hess.numpy()
    for i in range(10):
        ee_grad = gnp[i]
        Hee = hnp[i]
        
        ei0 = xnp[i, 0]
        ei1 = xnp[i, 1]
        ej0 = xnp[i, 2]
        ej1 = xnp[i, 3]
        gee_ipc = ipctk.edge_edge_distance_gradient(ei0, ei1, ej0, ej1)

        Hee_ipc = ipctk.edge_edge_distance_hessian(ei0, ei1, ej0, ej1)

        # print(f"ee_grad = {ee_grad}\nref = {gee_ipc}")
        print(f"diff grad = {np.linalg.norm(ee_grad - gee_ipc)}")
        print(f"H = {np.linalg.norm(Hee)}\nref = {np.linalg.norm(Hee_ipc)}, diff = {np.linalg.norm(Hee - Hee_ipc)}")
