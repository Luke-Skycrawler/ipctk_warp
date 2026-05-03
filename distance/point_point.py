import warp as wp 
from scalar_types import *
@wp.func 
def point_point_distance_gradient_hessian(p0: vec3, p1: vec3):
    '''
    already psd 
    '''
    d = p1 - p0
    grad = make_vec6(-d, d) * scalar(2.0)

    v = make_vec6(vec3(scalar(-1.0)), vec3(scalar(1.0)))

    hess = wp.outer(v, v) * scalar(2.0)
    return grad, hess