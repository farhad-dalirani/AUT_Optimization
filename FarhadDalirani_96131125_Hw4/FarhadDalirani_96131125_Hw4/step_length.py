from conditions import wolf_condition_1
import numpy as np


def back_tracking(func, func_gradient, xk, pk, A, P, b, q, t, initial_alpha=2, c1=0.0001, ru=0.95):
    """
    backtracking for finding acceptable alpha
    :param func:
    :param func_gradient:
    :param xk:
    :param pk:
    :param initial_alpha:
    :param c1:
    :param ru:
    :return:
    """
    if c1 >= 1 or c1 <= 0:
        raise ValueError('C1 must be in range (0,1)')
    if ru >= 1 or ru <= 0:
        raise ValueError('ru must be in range (0,1)')

    alpha = initial_alpha

    # check and reduce alpha until sufficient decrease is satisfied
    while wolf_condition_1(func=func, func_gradient=func_gradient, xk=xk, pk=pk, alpha=alpha, c1=c1, A=A, P=P, b=b, q=q, t=t) != True:
        # reduce alpha
        alpha = ru * alpha

    return alpha


def step_length_func(func, func_gradient, xk, pk, A, P, b, q, t, initial_alpha=2, c1=0.0001, ru=0.95):
    """
    Depends on mode, it choose back tracking or interpolation to find appropriate step length
    :param func:
    :param func_gradient:
    :param xk:
    :param pk:
    :param initial_alpha:
    :param c1:
    :param ru:
    :param mode:
    :return:
    """
    return back_tracking(func=func, func_gradient=func_gradient, xk=xk, pk=pk, initial_alpha=initial_alpha, c1=c1,
                         ru=ru, A=A, P=P, b=b, q=q, t=t)

