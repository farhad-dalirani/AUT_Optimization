import numpy as np
from phi import phi, phi_gradient, phi_hessian


def f0_barrier_without_phi(x, A, P, b, q):
    """
    function fn that is given in the homework 4
    :param x:
    :param A:
    :param P:
    :param b:
    :param q:
    :param t:
    :return:
    """
    if x.shape[0] != A.shape[1] or x.shape[1] != 1:
        raise ValueError('X must have dimension n*1')

    f0 = 0.5*np.dot(np.dot(np.transpose(x), P), x) + np.dot(np.transpose(q), x)

    return float(f0)


def f0_barrier(x, A, P, b, q, t):
    """
    function fn that is given in the homework 4
    :param x:
    :param A:
    :param P:
    :param b:
    :param q:
    :param t:
    :return:
    """
    if x.shape[0] != A.shape[1] or x.shape[1] != 1:
        raise ValueError('X must have dimension n*1')

    f0 = 0.5*np.dot(np.dot(np.transpose(x), P), x) + np.dot(np.transpose(q), x)
    phi_x = phi(x=x, A=A, b=b)

    # t * f0 + Phi
    f0_barrier_value = t*f0 + phi_x

    return float(f0_barrier_value)


def f0_barrier_gradient(x, A, P, b, q, t):
    """
    function fn that is given in the homework 4
    :param x:
    :param A:
    :param P:
    :param b:
    :param q:
    :param t:
    :return:
    """
    if x.shape[0] != A.shape[1] or x.shape[1] != 1:
        raise ValueError('X must have dimension n*1')

    f0_gradient = np.dot(P, x) + q
    phi_grad_x = phi_gradient(x=x, A=A, b=b)

    # t * f0_grad + Phi_grad
    f0_barrier_value_gradient = t*f0_gradient + phi_grad_x

    return f0_barrier_value_gradient


def f0_barrier_hessian(x, A, P, b, q, t):
    """
    Hessian of function fn that is given in the homework 4
    :param x:
    :param A:
    :param P:
    :param b:
    :param q:
    :param t:
    :return:
    """
    if x.shape[0] != A.shape[1] or x.shape[1] != 1:
        raise ValueError('X must have dimension n*1')

    f0_hessian = P
    phi_hessian_x = phi_hessian(x=x, A=A, b=b)

    # t * f0_grad + Phi_grad
    f0_barrier_value_gradient = t*f0_hessian + phi_hessian_x

    return f0_barrier_value_gradient


