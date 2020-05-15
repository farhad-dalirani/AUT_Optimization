from search_directions import newton_decent_directions
import numpy as np
from step_length import back_tracking
from sklearn.datasets import make_spd_matrix


def newton_descent_a_iteration(func, func_gradient, func_hessian, xk, A, P, b, q, t, initial_alpha=2.0, c1=0.0001, ru=0.95):
    """
    This function is implementation of on iteration of newton descent,
    :param func:
    :param func_gradient:
    :param xk:
    :return:
    """
    # newton descent direction
    pk = newton_decent_directions(function=func, func_derivative=func_gradient, func_hessian=func_hessian, xk=xk, A=A, P=P, b=b, q=q, t=t)

    alpha_k = None

    # newton descent step length by backtracking
    alpha_k = back_tracking(func=func, func_gradient=func_gradient, xk=xk, pk=pk,
                            initial_alpha=initial_alpha, c1=c1, ru=ru, A=A, P=P, b=b, q=q, t=t)

    # next point
    xk_plus_one = xk + alpha_k * pk

    return xk_plus_one, alpha_k, pk


def newton_descent(func, func_gradient, func_hessian, x0, A, P, b, q, t, max_iteration=500, initial_alpha=2.0, c1=0.0001, ru=0.95, auto_halt=True):
    """

    :param func:
    :param func_gradient:
    :param x0:
    :param max_iteration:
    :param initial_alpha:
    :param c1:
    :param ru:
    :return:
    """
    # generated sequence of xk,pk, alphak, f(xk)
    x_s= [x0]
    pks = []
    alpha_s = []
    f_s = [func(x=x_s[-1], A=A, P=P, b=b, q=q, t=t)]

    # for max allowed iterations
    for i in range(0, max_iteration):
        # find direction, length, xk+1
        xk_plus_one, alpha_k, pk = newton_descent_a_iteration(func=func, func_gradient=func_gradient, func_hessian=func_hessian,
                                                              xk=x_s[-1], initial_alpha=initial_alpha, c1=c1, ru=ru, A=A, P=P, b=b, q=q, t=t)
        x_s.append(xk_plus_one)
        pks.append(pk)
        alpha_s.append(alpha_k)
        f_s.append(func(x=xk_plus_one, A=A, P=P, b=b, q=q, t=t))

        if len(f_s) > 60 and auto_halt==True:
            slow_decrease = True
            for j in range(-1, -8, -1):
                if abs(float(f_s[j])-float(f_s[j-1])) > 0.0001:
                    slow_decrease = False
                    break
            if slow_decrease == True:
                break
        #gradient_x = func_gradient(x=x_s[-1], A=A, P=P, b=b, q=q, t=t)
        #hessian_x = func_hessian(x=x_s[-1], A=A, P=P, b=b, q=q, t=t)

        #lambda_2_2 = np.dot(np.transpose(gradient_x), np.dot(np.linalg.inv(hessian_x), gradient_x))/2.0
        #if lambda_2_2 <= 0.01:
        #    break

    return x_s, f_s, alpha_s, pks

