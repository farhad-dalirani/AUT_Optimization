from newton_descent import newton_descent
import numpy as np
from sklearn.datasets import make_spd_matrix


def barrier_method(func, func_gradient, func_hessian, x0, A, P, b, q, t0=0.0001, max_iteration=1000, epsilon=0.0001, micro=50):
    """
    Barrier Method
    :param func:
    :param func_gradient:
    :param func_hessian:
    :param x0:
    :param A:
    :param P:
    :param b:
    :param q:
    :param t0:
    :param max_iteration:
    :return:
    """
    t = t0
    x = np.copy(x0)
    m = A.shape[0]
    iter = 0

    x_s_barrier = []
    f_s_barrier = []
    duality_gap = []

    while True:
        # solve t f0 + phi by newton method
        x_s, f_s, alpha_s, pks = newton_descent(func=func, func_gradient=func_gradient, func_hessian=func_hessian,
                                                x0=x, max_iteration=3000, initial_alpha=0.003, A=A, P=P, b=b, q=q, t=t)
        x_s_barrier.append(x_s)
        f_s_barrier.append(f_s)
        duality_gap.append(m/t)

        # update x
        x = np.copy(x_s[-1])
        # halt
        if (m/t < epsilon) or iter > max_iteration:
            break
        # increase t
        t = t * micro
        iter = iter + 1

    return x_s_barrier, f_s_barrier, duality_gap


#############################################################
#           This part is for  running barrier method
#############################################################
if __name__ == '__main__':
    from functions_derivatives import *
    from plot_function import *

    # for five different set of A, b, P and q solve:
    # minimize      (1/2) (trans x)Px + (trans q)x
    # subject to    Ax <= b
    for i in range(0, 5):
        # different seed for generating different values of A, b, P and q
        np.random.seed(i)
        x0 = np.zeros((50,1))
        A = np.float32(np.random.randint(low=0, high=20, size=[100,50]))
        b = np.float32(np.random.randint(low=150, high=200, size=[100,1]))
        P = make_spd_matrix(n_dim=50)
        q = np.float32(np.random.randint(low=0, high=20, size=[50,1]))

        # with back tracking
        x_barrier_s, f_barrier_s, duality_gap = barrier_method(func=f0_barrier, func_gradient=f0_barrier_gradient, func_hessian=f0_barrier_hessian,
                                                x0=x0, A=A, P=P, b=b, q=q, t0=1, max_iteration=1000, epsilon=0.00001, micro=50)
        print('X:\n\n',x_barrier_s, '\n\ntf+phi:\n', f_barrier_s, '\n\nDuality Gap:\n', duality_gap, '\n********************\n')
        plot_func_in_iterations_barrier(func=f0_barrier_without_phi, x_barrier_s=x_barrier_s, A=A, P=P, b=b, q=q)
        plot_duality_gap_barrier(x_barrier_s=x_barrier_s, duality_gaps=duality_gap)

    plt.show()