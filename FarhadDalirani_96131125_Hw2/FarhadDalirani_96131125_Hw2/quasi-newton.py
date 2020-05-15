from step_length import back_tracking, interpolation
from search_directions import quasi_newton_direction
import numpy as np


def quasi_newton_a_iteration(func, func_gradient, func_hessian, xk, xk_plus_1, Bk, initial_alpha=2.0, c1=0.0001, ru=0.95, step_len='backtracking'):
    """
    This function is implementation of on iteration of quasi newton,
    :param func:
    :param func_gradient:
    :param xk:
    :return:
    """
    # quasi newton direction
    Bk_plus_1, pk = quasi_newton_direction(func_derivative=func_gradient, func_hessian=func_hessian, xk=xk, xk_plus_1=xk_plus_1, Bk=Bk)

    if step_len == 'backtracking':
        # quasi newton step length by backtracking
        alpha_k = back_tracking(func=func, func_gradient=func_gradient, xk=xk_plus_1, pk=pk,
                            initial_alpha=initial_alpha, c1=c1, ru=ru)
    elif step_len == 'interpolation':
        alpha_k = interpolation(func=func, func_gradient=func_gradient, xk=xk_plus_1, pk=pk,
                                initial_alpha=initial_alpha, c1=c1)
    else:
        raise ValueError('step_len value is not valid!')
    # next point
    xk_plus_one = xk_plus_1 + alpha_k * pk

    return xk_plus_one, alpha_k, Bk_plus_1, pk


def quasi_newton(func, func_gradient, func_hessian, x0, max_iteration=1000, initial_alpha=2.0, c1=0.0001, ru=0.95, auto_halt=True, step_len='backtracking'):
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
    f_s = [func(x_s[-1])]
    bk_s = [None]

    # for max allowed iterations
    for i in range(0, max_iteration):
        # find direction, length, xk+1
        if i != 0:
            xk_plus_one, alpha_k, Bk_plus_1, pk = quasi_newton_a_iteration(func=func, func_gradient=func_gradient, func_hessian=func_hessian
                                                            , xk=x_s[-2], xk_plus_1=x_s[-1], Bk=bk_s[-1], initial_alpha=initial_alpha,
                                                            c1=c1, ru=ru,step_len=step_len)
        else:
            xk_plus_one, alpha_k, Bk_plus_1, pk = quasi_newton_a_iteration(func=func, func_gradient=func_gradient,
                                                                           func_hessian=func_hessian
                                                                           , xk=None, xk_plus_1=x_s[-1], Bk=bk_s[-1],
                                                                           initial_alpha=initial_alpha,
                                                                           c1=c1, ru=ru, step_len=step_len)

        x_s.append(xk_plus_one)
        pks.append(pk)
        alpha_s.append(alpha_k)
        f_s.append(func(xk_plus_one))
        bk_s.append(Bk_plus_1)

        if f_s[-1] == 0.0:
            break
        if len(f_s) > 50 and auto_halt==True:
            slow_decrease = True
            for j in range(-1, -8, -1):
                if float(f_s[j])/float(f_s[j-1]) < 0.9999:
                    slow_decrease = False
                    break
            if slow_decrease == True:
                break

    return x_s, f_s, alpha_s, bk_s, pks



#############################################################
#           This part is for  running Quasi Newton
#############################################################
if __name__ == '__main__':
    from functions_derivatives import *
    from plot_function import *
    from phi import plot_phi

    #x0 = np.array([2, 1])
    x0 = np.array([0, 1])
    #x0 = np.array([-1.01, 1.01])
    #x0 = np.array([-1.5,3])
    #x0 = np.array([2.04, 1.03])
    #x0 = np.array([5, 5])
    #x0 = np.array([0, 7])
    #x0 = np.array([20, 20])
    #x0 = np.array([0.0125, -0.25])
    x0 = np.reshape(x0, newshape=[2,1])

    x_s, f_s, alpha_s, bks, pks = quasi_newton(func=f1, func_gradient=f1_gradient, func_hessian=f1_hessian, x0=x0,
                                               max_iteration=1000, initial_alpha=0.2, step_len='backtracking')
    #x_s, f_s, alpha_s, bks, pks = quasi_newton(func=f1, func_gradient=f1_gradient, func_hessian=f1_hessian, x0=x0,
    #                                           max_iteration=1000, initial_alpha=0.3, step_len='interpolation')

    # plot for first iteration
    plot_phi(func=f1, func_gradient=f1_gradient, xk=x_s[0], pk=pks[0], c1=0.0001, c2=0.9, ru=0.95, initial_alpha=0.2,
             condition='wolfe',
             description='f1, first iteration, wolfe', offset=0.65)
    plot_phi(func=f1, func_gradient=f1_gradient, xk=x_s[-1], pk=pks[-1], c1=0.0001, c2=0.9, ru=0.95, initial_alpha=0.2,
             condition='wolfe',
             description='f1, last iteration, wolfe', offset=1)
    plot_phi(func=f1, func_gradient=f1_gradient, xk=x_s[0], pk=pks[0], c1=0.0001, c2=0.9, ru=0.95, initial_alpha=0.2,
             condition='golden',
             description='f1, first iteration, golden', offset=2.6)
    plot_phi(func=f1, func_gradient=f1_gradient, xk=x_s[-1], pk=pks[-1], c1=0.0001, c2=0.9, ru=0.95, initial_alpha=0.2,
             condition='golden',
             description='f1, last iteration, golden', offset=2)

    print(x_s)
    print(f_s)
    plot_function_with_trajectory(func=f1, mode='3D', xs=x_s, fs=f_s)
    plot_func_in_iterations(fs=f_s)
    plot_decrease_in_iterations(fs=f_s)
    plot_norm_pk_in_iterations(pk_s=pks)

    print('===============================================')
    x0 = np.array([[2]])
    #x0 = np.array([[-1]])
    #x0 = np.array([[4]])

    x_s, f_s, alpha_s, bks, pks = quasi_newton(func=f2, func_gradient=f2_gradient, func_hessian=f2_hessian, x0=x0,
                                              max_iteration=1000, initial_alpha=0.2, step_len='backtracking')
    #x_s, f_s, alpha_s, bks, pks = quasi_newton(func=f2, func_gradient=f2_gradient, func_hessian=f2_hessian, x0=x0,
    #                                           max_iteration=1000, initial_alpha=0.2, step_len='interpolation')

    # plot for first iteration
    plot_phi(func=f2, func_gradient=f2_gradient, xk=x_s[0], pk=pks[0], c1=0.0001, c2=0.9, ru=0.95, initial_alpha=0.2,
             condition='wolfe',
             description='f2, first iteration, wolfe', offset=0.65)
    plot_phi(func=f2, func_gradient=f2_gradient, xk=x_s[-1], pk=pks[-1], c1=0.0001, c2=0.9, ru=0.95, initial_alpha=0.2,
             condition='wolfe',
             description='f2, last iteration, wolfe', offset=1)
    plot_phi(func=f2, func_gradient=f2_gradient, xk=x_s[0], pk=pks[0], c1=0.0001, c2=0.9, ru=0.95, initial_alpha=0.2,
             condition='golden',
             description='f2, first iteration, golden', offset=0.6)
    plot_phi(func=f2, func_gradient=f2_gradient, xk=x_s[-1], pk=pks[-1], c1=0.0001, c2=0.9, ru=0.95, initial_alpha=0.2,
             condition='golden',
             description='f2, last iteration, golden', offset=2)

    print(x_s)
    print(f_s)
    plot_function_with_trajectory(func=f2, mode='2D', xs=x_s, fs=f_s)
    plot_func_in_iterations(fs=f_s)
    plot_decrease_in_iterations(fs=f_s)
    plot_norm_pk_in_iterations(pk_s=pks)

    plt.show()