from conditions import wolf_condition_1
import numpy as np


def back_tracking(func, func_gradient, xk, pk, initial_alpha=2, c1=0.0001, ru=0.95):
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
    while wolf_condition_1(func=func, func_gradient=func_gradient, xk=xk, pk=pk, alpha=alpha, c1=c1) != True:
        # reduce alpha
        alpha = ru * alpha

    return alpha


def interpolation(func, func_gradient, xk, pk, initial_alpha=2, c1=0.0001):
    """
    interpolation for finding acceptable alpha
    :param func:
    :param func_gradient:
    :param xk:
    :param pk:
    :param initial_alpha:
    :param c1:
    :param ru:
    :return:
    """
    from phi import phi, phi_derivative_alpha_zero

    if c1 >= 1 or c1 <= 0:
        raise ValueError('C1 must be in range (0,1)')

    alpha_0 = initial_alpha

    # quadratic approximation
    alpha_1_numerator = (phi_derivative_alpha_zero(func_derivative=func_gradient, xk=xk, pk=pk)*(alpha_0**2))
    alpha_1_denominator = 2*(phi(func=func, xk=xk, pk=pk, alpha=alpha_0)-phi(func=func, xk=xk, pk=pk, alpha=0)-phi_derivative_alpha_zero(func_derivative=func_gradient, xk=xk, pk=pk)*(alpha_0))

    if alpha_1_denominator != 0:
        alpha_1 = - alpha_1_numerator/alpha_1_denominator
    else:
        alpha_1 = - alpha_1_numerator / (alpha_1_denominator+0.000001)

    if wolf_condition_1(func=func, func_gradient=func_gradient, xk=xk, pk=pk, alpha=alpha_1, c1=c1) == True:
        return alpha_1

    # cubic approximation
    while True:
        coefficient = 1.0/((alpha_0**2)*(alpha_1**2)*(alpha_1-alpha_0))
        coefficient = float(coefficient)
        matrix_1 = np.array([[alpha_0**2, -(alpha_1**2)], [-(alpha_0**3), alpha_1**3]])
        matrix_2 = np.array([[phi(func=func, xk=xk, pk=pk, alpha=alpha_1)-phi(func=func, xk=xk, pk=pk, alpha=0)-phi_derivative_alpha_zero(func_derivative=func_gradient, xk=xk, pk=pk)*(alpha_1)],
                             [phi(func=func, xk=xk, pk=pk, alpha=alpha_0)-phi(func=func, xk=xk, pk=pk, alpha=0)-phi_derivative_alpha_zero(func_derivative=func_gradient, xk=xk, pk=pk)*(alpha_0)]])
        product = coefficient * np.dot(matrix_1, matrix_2)
        a = product[0, 0]
        b = product[1, 0]

        #temp1 = (-b+np.sqrt((b**2)-3*a*phi_derivative_alpha_zero(func_derivative=func_gradient, xk=xk, pk=pk)))
        #temp2 = -3*a*phi_derivative_alpha_zero(func_derivative=func_gradient, xk=xk, pk=pk)
        #temp3 = (b**2)-3*a*phi_derivative_alpha_zero(func_derivative=func_gradient, xk=xk, pk=pk)
        #temp6 = a

        alpha_2 = (-b+np.sqrt((b**2)-3*a*phi_derivative_alpha_zero(func_derivative=func_gradient, xk=xk, pk=pk)))/(3*a)

        if wolf_condition_1(func=func, func_gradient=func_gradient, xk=xk, pk=pk, alpha=alpha_2, c1=c1) == True:
            return alpha_2

        alpha_0 = alpha_1
        alpha_1 = alpha_2


def step_length_func(func, func_gradient, xk, pk, initial_alpha=2, c1=0.0001, ru=0.95, mode='backtracking'):
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
    if mode == 'backtracking':
        return back_tracking(func=func, func_gradient=func_gradient, xk=xk, pk=pk, initial_alpha=initial_alpha, c1=c1, ru=ru)
    elif mode == 'interpolation':
        return interpolation(func=func, func_gradient=func_gradient, xk=xk, pk=pk, initial_alpha=initial_alpha, c1=c1)
    else:
        raise ValueError('mode is not valid!')

