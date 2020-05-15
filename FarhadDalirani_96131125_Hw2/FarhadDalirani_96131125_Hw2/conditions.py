import numpy as np


def wolf_conditions(func, func_gradient, xk, pk, alpha, c1=0.0001, c2=0.9):
    """
    Check wolf conditions
    :param func:
    :param func_gradient:
    :param xk:
    :param pk:
    :param alpha:
    :param c1:
    :param c2:
    :return:
    """
    if c1 >= 1 or c1 <= 0:
        raise ValueError('C1 must be in range (0,1)')

    if c2 >= 1 or c2 <= 0:
        raise ValueError('C2 must be in range (0,1)')

    if c2 <= c1 or c2 >= 1:
        raise ValueError('C2 must be in range (c1,1)')

    if wolf_condition_1(func=func, func_gradient=func_gradient, xk=xk, pk=pk, alpha=alpha, c1=c1) and\
            wolf_condition_2(func_gradient=func_gradient, xk=xk, pk=pk, alpha=alpha, c2=c2):
        return True
    else:
        return False


def wolf_condition_1(func, func_gradient, xk, pk, alpha, c1):
    """
    Check wolf conditions 1 - sufficient decrease
    :param func:
    :param func_gradient:
    :param xk:
    :param pk:
    :param alpha:
    :param c1:
    :return:
    """
    if c1 >= 1 or c1 <= 0:
        raise ValueError('C1 must be in range (0,1)')

    # left side of sufficient decrease
    left_side = func(xk+ alpha * pk)
    left_side = float(left_side)

    # right side of sufficient decrease
    right_side = func(xk) + c1 * alpha * np.dot(np.transpose(func_gradient(xk)), pk)
    right_side = float(right_side)

    if left_side <= right_side:
        return True
    else:
        return False


def wolf_condition_2(func_gradient, xk, pk, alpha, c2):
    """
    Check wolf conditions 2 - curvature
    :param func_gradient:
    :param xk:
    :param pk:
    :param alpha:
    :param c2:
    :return:
    """
    if c2 >= 1 or c2 <= 0:
        raise ValueError('C2 must be in range (0,1)')

    # left side of curvature
    left_side = np.dot(np.transpose(func_gradient(xk+alpha*pk)), pk)

    # right side of curvature
    right_side = c2 * np.dot(np.transpose(func_gradient(xk)), pk)

    if left_side >= right_side:
        return True
    else:
        return False


def goldstein(func, func_gradient, xk, pk, alpha, c1=0.0001):
    """
    Check goldstein conditions
    :param func:
    :param func_gradient:
    :param xk:
    :param pk:
    :param alpha:
    :param c1:
    :return:
    """
    if c1 >= 1 or c1 <= 0:
        raise ValueError('C1 must be in range (0,1)')

    # left side of sufficient decrease
    mid_side = func(xk+ alpha * pk)
    mid_side = float(mid_side)

    # right side of goldstein
    right_side = func(xk) + c1 * alpha * np.dot(np.transpose(func_gradient(xk)), pk)
    right_side = float(right_side)

    # left side of goldstein
    left_side = func(xk) + (1-c1) * alpha * np.dot(np.transpose(func_gradient(xk)), pk)
    left_side = float(left_side)

    if mid_side <= right_side and left_side <= mid_side:
        return True
    else:
        return False
