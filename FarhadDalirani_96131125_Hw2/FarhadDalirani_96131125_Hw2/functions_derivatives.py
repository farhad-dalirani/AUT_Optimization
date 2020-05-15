import numpy as np


def f1(x):
    """
    function f1 that is given in the homework 2
    :param x: must be a 2*1 numpy array
    :return:
    """
    if x.shape[0] != 2 or x.shape[1] != 1:
        raise ValueError('X must have dimension 2*1')

    x1 = x[0,0]
    x2 = x[1,0]
    return 100*((x2 - (x1**2))**2) + ((1-x1)**2)


def f1_gradient(x):
    """
    gradient of function f1
    :param x: must be a 2*1 numpy array
    :return:
    """
    if x.shape[0] != 2 or x.shape[1] != 1:
        raise ValueError('X must have dimension 2*1')

    x1 = x[0, 0]
    x2 = x[1, 0]

    df_dx1 = -400 * (x2-(x1**2)) * (x1) - 2 * (1-x1)
    df_dx2 = 200 * (x2 - (x1**2))

    gradient = np.zeros(shape=[2,1])
    gradient[0, 0] = df_dx1
    gradient[1, 0] = df_dx2

    return gradient


def f1_hessian(x):
    """
    Hessian of function f1
    :param x: must be a 2*1 numpy array
    :return:
    """
    if x.shape[0] != 2 or x.shape[1] != 1:
        raise ValueError('X must have dimension 2*1')

    x1 = x[0, 0]
    x2 = x[1, 0]

    hessian = [[2 + 800 * (x1**2) - 400*(-x1**2 + x2), -400*x1], [-400*x1, 200]]
    hessian = np.array(hessian)

    return hessian


def f2(x):
    """
    function f2 that is given in the homework 2
    :param x: must be a scalar
    :return:
    """
    return x**2 + np.exp((-1)/((100*(x-1))**2)) - 1


def f2_gradient(x):
    """
    gradient of function f2
    :param x: must be a scalar
    :return:
    """
    gradient = 2*x + (1/(5000*((x-1)**3)))*(np.exp(-1/((100*(x-1))**2)))
    gradient = np.array(gradient)
    gradient = np.reshape(gradient, newshape=(1,1))
    return gradient


def f2_hessian(x):
    """
    hessian of function f2
    :param x: must be a scalar
    :return:
    """
    hessian = np.exp(-1/(10000 * (x - 1)**2)) * (1 - 15000 * ((x - 1)**2))/(25000000 * ((x - 1)**6)) + 2
    hessian = np.array(hessian)
    hessian = np.reshape(hessian, newshape=(1,1))
    return hessian



