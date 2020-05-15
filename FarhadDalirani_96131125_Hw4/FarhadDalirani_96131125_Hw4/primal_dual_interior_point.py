import numpy as np
from primal_dual_functions_drivatives import *
from sklearn.datasets import make_spd_matrix


def compute_primal_dual_direction(x, A, P, b, q, t, lambdas):
    """
    Compute delta_y
    :param x:
    :param A:
    :param P:
    :param b:
    :param q:
    :param t:
    :param lamdas
    :return:
    """
    # r=(r_dual,r_cent,r_pri)
    # dr is a 3*3 matrix
    # There is no equality constrain so dr is a 2*2 matrix
    dr11 = f0_hessian(x=x, A=A, P=P, b=b, q=q)
    dr12 = np.transpose(f_derivative(x=x, A=A, P=P, b=b, q=q))
    dr21 = -1 * np.dot(np.diag(lambdas.T.tolist()[0]), f_derivative(x=x, A=A, P=P, b=b, q=q))
    dr22 = -1 * np.diag(f(x=x, A=A, P=P, b=b, q=q).T.tolist()[0])

    dr11_12 = np.hstack((dr11, dr12))
    dr21_22 = np.hstack((dr21, dr22))

    dr = np.vstack((dr11_12,dr21_22))

    r_dual_value = r_dual(x=x, A=A, P=P, b=b, q=q, lambdas=lambdas)
    r_cent_value = r_cent(x=x, A=A, P=P, b=b, q=q, lambdas=lambdas, t=t)

    r = np.vstack((r_dual_value, r_cent_value))

    # solve [dr * delta_y = -r] by delta_y = -inv(Dr)r
    #delta_y = -1 * np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(dr), dr)), np.transpose(dr)), r)
    delta_y = -1 * np.dot(np.linalg.inv(dr), r)

    delta_x = delta_y[0:x.shape[0], 0:1]
    delta_lambda = delta_y[x.shape[0]::, 0:1]

    return delta_x, delta_lambda


def backtracking(func, x, A, P, b, q, lambdas, delta_x, delta_lambdas, t, alpha=0.1, beta = 0.8):
    """
    Backtracking for primal-dual interior point
    :param func:
    :param x:
    :param A:
    :param P:
    :param b:
    :param q:
    :param lambdas:
    :param delta_x:
    :param delta_lambdas:
    :param alpha:
    :param beta:
    :return:
    """
    r_dual_value = r_dual(x=x, A=A, P=P, b=b, q=q, lambdas=lambdas)
    r_cent_value = r_cent(x=x, A=A, P=P, b=b, q=q, lambdas=lambdas, t=t)
    r = np.vstack((r_dual_value, r_cent_value))

    # find smax
    temp = np.inf
    for i in range(0, A.shape[0]):
       if delta_lambdas[i,0] < 0:
           temp = min(temp, -lambdas[i]/delta_lambdas[i])
    smax = min(temp, 1)
    #######################
    #smax = 1

    s = 0.99*smax
    while True:
        x_next = x + s * delta_x
        lambdas_next = lambdas + s * delta_lambdas

        f_next = f(x=x_next, A=A, P=P, b=b, q=q)
        if np.all(f_next < 0):
            if np.all(lambdas_next > 0):
                r_dual_value = r_dual(x=x_next, A=A, P=P, b=b, q=q, lambdas=lambdas_next)
                r_cent_value = r_cent(x=x_next, A=A, P=P, b=b, q=q, lambdas=lambdas_next, t=t)
                r_next = np.vstack((r_dual_value, r_cent_value))

                if np.linalg.norm(r_next) <= (1-alpha*s)*np.linalg.norm(r):
                    return s

        s = beta * s


def primal_dual_interior_point_method(x0, A, P, b, q, lambdas0, micro=10, e_feas=0.001, e=0.001):
    """
    Primal Dual Algorithm
    :param x0:
    :param A:
    :param P:
    :param b:
    :param q:
    :param lambdas0:
    :param micro:
    :param e_feas:
    :param e:
    :return:
    """

    lambdas = np.copy(lambdas0)
    x = np.copy(x0)

    x_s = [x0]
    f_s = [f0(x=x0, A=A, P=P, b=b, q=q)]
    surrogate_duality_gaps = [surrogate_duality_gap(x=x0, A=A, P=P, b=b, q=q, lambdas=lambdas)]
    r_fea_s = [np.linalg.norm(r_dual(x=x0, A=A, P=P, b=b, q=q, lambdas=lambdas))]

    while True:
        # determine t
        t = micro * A.shape[0]/surrogate_duality_gap(x=x, A=A, P=P, b=b, q=q, lambdas=lambdas)

        #compute search direction
        delta_x, delta_lambdas = compute_primal_dual_direction(x=x, A=A, P=P, b=b, q=q, t=t, lambdas=lambdas)

        # line search and update
        s = backtracking(func=f0, x=x, A=A, P=P, b=b, q=q, lambdas=lambdas, delta_x=delta_x,
                         delta_lambdas=delta_lambdas, t=t, alpha=0.01, beta=0.5)
        x = x + s * delta_x
        lambdas = lambdas + s * delta_lambdas

        x_s.append(x)
        f_s.append(f0(x=x, A=A, P=P, b=b, q=q))
        surrogate_duality_gaps.append(surrogate_duality_gap(x=x, A=A, P=P, b=b, q=q, lambdas=lambdas))
        r_fea_s.append(np.linalg.norm(r_dual(x=x, A=A, P=P, b=b, q=q, lambdas=lambdas)))

        # halt
        if np.linalg.norm(r_dual(x=x, A=A, P=P, b=b, q=q, lambdas=lambdas)) <= e_feas and surrogate_duality_gap(x=x, A=A, P=P, b=b, q=q, lambdas=lambdas) <= e:
            break

    return x_s, f_s, surrogate_duality_gaps, r_fea_s


# test
if __name__ == '__main__':
    from primal_dual_functions_drivatives import *
    from plot_function import *

    # for five different set of A, b, P and q solve:
    # minimize      (1/2) (trans x)Px + (trans q)x
    # subject to    Ax <= b
    # different seed for generating different values of A, b, P and q
    for i in range(5):
        np.random.seed(i)
        x0 = np.zeros((50, 1))
        A = np.float32(np.random.randint(low=0, high=20, size=[100, 50]))
        b = np.float32(np.random.randint(low=150, high=200
                                         , size=[100, 1]))
        P = make_spd_matrix(n_dim=50)
        q = np.float32(np.random.randint(low=0, high=20, size=[50, 1]))
        lambdas0 = np.float32(np.random.randint(low=5, high=10, size=[100, 1]))

        # with back tracking
        x_interior_s, f_interior_s, surrogate_duality_gaps, r_fea_s = primal_dual_interior_point_method(x0=x0, A=A, P=P, b=b, q=q,
                                                                                                        lambdas0=lambdas0, micro=10, e_feas=0.0000001, e=0.00000001)

        print('X:\n\n', x_interior_s, '\n\nf:\n', f_interior_s, '\n\nSurrogate Duality Gap:\n', surrogate_duality_gaps, '\n\nr feas\n', r_fea_s,
              '\n********************\n')

        plot_func_in_iterations_interior(f_interior_s=f_interior_s, title='f(Xk)')
        plot_surrogate_duality_gap_interior(surrogate_duality_gaps=surrogate_duality_gaps, title='surrogate duality gaps')
        plot_dual_residual_interior(r_fea_s=r_fea_s, title='Dual Residual')

    plt.show()