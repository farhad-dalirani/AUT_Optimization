import numpy as np


def phi(x, A, b):
    """
    Phi in barrier method
    :param u:
    :param A:
    :param P:
    :param b:
    :param q:
    :param t:
    :return:
    """
    Fi_s = np.dot(A, x) - b
    neg_Fi_s = -Fi_s
    log_neg_Fi_s = np.log(neg_Fi_s)
    sum_log_neg_Fi_s = np.sum(log_neg_Fi_s)
    phi_value = -sum_log_neg_Fi_s

    return phi_value


def phi_gradient(x, A, b):
    """
    gradient of Phi in barrier method
    :param u:
    :param A:
    :param P:
    :param b:
    :param q:
    :param t:
    :return:
    """
    Fi_s = np.dot(A, x) - b
    neg_Fi_s = -Fi_s
    grad_Fi = np.transpose(A)
    value = grad_Fi/np.transpose(neg_Fi_s)

    grad_phi = np.sum(value,axis=1)
    grad_phi = np.reshape(grad_phi, newshape=(-1,1))

    return grad_phi


def phi_hessian(x, A, b):
    """
    Hessian of Phi in barrier method
    :param u:
    :param A:
    :param P:
    :param b:
    :param q:
    :param t:
    :return:
    """
    Fi_s = np.dot(A, x) - b
    grad_Fi = np.transpose(A)

    phi_hessian_value = np.zeros((grad_Fi.shape[0], grad_Fi.shape[0]))
    for i in range(0, grad_Fi.shape[1]):
        phi_hessian_value = phi_hessian_value + (np.dot(grad_Fi[:, i:i+1], np.transpose(grad_Fi[:, i:i+1])))/(Fi_s[i,0]*Fi_s[i,0])
    return phi_hessian_value


