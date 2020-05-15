import numpy as np
from trust_region_p import cauchy_point, dogleg


def m_k(func, func_gradient, func_hessian, Bk, xk, pk):
    """
    Quadratic Approximation of f in xk
    :param func:
    :param func_gradient:
    :param func_hessian:
    :param Bk:
    :param xk:
    :param pk:
    :return:
    """
    out = func(x=xk) + np.dot(np.transpose(func_gradient(x=xk)), pk) + 0.5 * np.dot(np.transpose(pk), np.dot(Bk, pk))
    return out


def trust_region(func, func_gradient, func_hessian, x0, delta_0=1.0, delta_max=2.0, mu=0.2, max_iteration=1000, auto_halt=True, method='cauchy'):
    """
    Trust region method for finding optimal point and optimal value
    :param func:
    :param func_gradient:
    :param x0:
    :param max_iteration:
    :param initial_alpha:
    :param c1:
    :param ru:
    :return:
    """
    if mu < 0 or mu >= 0.25:
        raise ValueError('Value of mu is not acceptable!')

    # generated sequence of xk,pk, f(xk)
    x_s= [x0]
    Bk_s = []
    gk_s = []
    pks = []
    f_s = [func(x_s[-1])]
    delta_s = [delta_0]
    ru_s = []

    # for max allowed iterations
    for i in range(0, max_iteration):

        #gk
        gk_s.append(func_gradient(x=x_s[-1]))
        Bk_s.append(func_hessian(x=x_s[-1]))

        # find direction
        pk = None
        if method == 'cauchy':
            pk = cauchy_point(gk=gk_s[-1], Bk=Bk_s[-1], deltak=delta_s[-1])
        elif method == 'dogleg':
            pk = dogleg(gk=gk_s[-1], Bk=Bk_s[-1], deltak=delta_s[-1])
        else:
            raise ValueError('Method for trust region is not valid!')

        if pk is None:
            raise ValueError('Returned pk in None!')

        # add new pk to previous pks list
        pks.append(pk)

        # calculate ru_k
        ru_k_enumerator = (func(x=x_s[-1])-func(x=x_s[-1]+pk))
        ru_k_denumerator = (m_k(func=func, func_gradient=func_gradient, func_hessian=func_hessian, Bk=Bk_s[-1], xk=x_s[-1], pk=pk*0)-
                m_k(func=func, func_gradient=func_gradient, func_hessian=func_hessian, Bk=Bk_s[-1], xk=x_s[-1], pk=pk))

        if ru_k_denumerator != 0:
            ru_k = ru_k_enumerator/ru_k_denumerator
        else:
            ru_k = ru_k_enumerator / (ru_k_denumerator+0.0000001)

        ru_k = float(ru_k)
        ru_s.append(ru_k)

        if ru_k < 0.25:
            delta_s.append(0.25*delta_s[-1])
        else:
            if ru_k > (3.0/4) and np.linalg.norm(pk, 2)==delta_s[-1]:
                delta_s.append(min(delta_max, 2 * delta_s[-1]))
            else:
                delta_s.append(delta_s[-1])

        if ru_k > mu:
            x_s.append(x_s[-1]+pk)
        else:
            x_s.append(x_s[-1])

        f_s.append(func(x_s[-1]))

        if f_s[-1] == 0.0:
            break
        if len(f_s) > 50 and auto_halt == True:
            slow_decrease = True
            for j in range(-1, -8, -1):
                if float(f_s[j])/float(f_s[j-1]) < 0.9999:
                    slow_decrease = False
                    break
            if slow_decrease == True:
                break

    return x_s, f_s, delta_s, pks, ru_s


#############################################################
#           This part is for  running Trust Region
#############################################################
if __name__ == '__main__':
    from functions_derivatives import *
    from plot_function import *
    from phi import plot_phi

    x0 = np.array([2, 1])
    # x0 = np.array([0, 1])
    # x0 = np.array([-1, 1])
    #x0 = np.array([-1.5,3])
    #x0 = np.array([2.04, 1.03])
    #x0 = np.array([5, 5])
    #x0 = np.array([0, 7])
    #x0 = np.array([20, 20])
    x0 = np.reshape(x0, newshape=[2,1])

    #x_s, f_s, delta_s, pks, ru_s = trust_region(func=f1, func_gradient=f1_gradient, func_hessian=f1_hessian, delta_0=1,
    #                                      delta_max=2, mu=0.2, x0=x0, max_iteration=1000, method='cauchy')
    x_s, f_s, delta_s, pks, ru_s = trust_region(func=f1, func_gradient=f1_gradient, func_hessian=f1_hessian, delta_0=1,
                                          delta_max=2, mu=0.2, x0=x0, max_iteration=1000, method='dogleg')


    print(x_s)
    print(f_s)
    plot_function_with_trajectory(func=f1, mode='3D', xs=x_s, fs=f_s)
    plot_func_in_iterations(fs=f_s)
    plot_ru_in_iterations(ru_s=ru_s)
    plot_decrease_in_iterations(fs=f_s)


    print('===============================================')
    #x0 = np.array([[2]])
    x0 = np.array([[-1]])
    #x0 = np.array([[4]])

    x_s, f_s, delta_s, pks, ru_s = trust_region(func=f2, func_gradient=f2_gradient, func_hessian=f2_hessian, delta_0=1,
                                          delta_max=2, mu=0.2, x0=x0, max_iteration=1000, method='cauchy')
    #x_s, f_s, delta_s, pks, ru_s = trust_region(func=f2, func_gradient=f2_gradient, func_hessian=f2_hessian, delta_0=1,
    #                                            delta_max=2, mu=0.2, x0=x0, max_iteration=1000, method='dogleg')

    print(x_s)
    print(f_s)
    plot_function_with_trajectory(func=f2, mode='2D', xs=x_s, fs=f_s)
    plot_func_in_iterations(fs=f_s)
    plot_ru_in_iterations(ru_s=ru_s)
    plot_decrease_in_iterations(fs=f_s)

    plt.show()
