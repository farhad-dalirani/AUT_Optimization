import numpy as np
from step_length import back_tracking, interpolation
import matplotlib.pyplot as plt

def phi(func, xk, pk, alpha):
    """
    phi(alpha) = func(xk+alpha * pk)
    :param xk:
    :param pk:
    :param alpha:
    :return:
    """
    # calculate phi alpha
    phi_alpha = func(xk + alpha * pk)

    return float(phi_alpha)


def phi_derivative_alpha_zero(func_derivative, xk, pk):
    """
    derivative of phi(0)
    :param func:
    :param xk:
    :param pk:
    :param alpha:
    :return:
    """
    derivative_phi_alpha_zero = np.dot(np.transpose(func_derivative(xk)), pk)

    return float(derivative_phi_alpha_zero)


def plot_phi(func, func_gradient, xk, pk, c1=0.0001, c2=0.9, ru=0.95, initial_alpha=2.0, condition='wolfe', description='', offset=1.0, method='backtrack'):
    """
    plot phi for specific xk and pk,
    :param func:
    :param func_derivative:
    :param xk:
    :param pk:
    :param alpha:
    :param condistion:
    :return:
    """
    from conditions import wolf_conditions, goldstein

    # find appropriate value for alpha
    if method == 'backtrack':
        # steepest descent step length by backtracking
        alpha_k = back_tracking(func=func, func_gradient=func_gradient, xk=xk, pk=pk,
                            initial_alpha=initial_alpha, c1=c1, ru=ru)
    elif method == 'interpolation':
        alpha_k = interpolation(func=func,func_gradient=func_gradient,xk=xk,pk=pk,initial_alpha=initial_alpha,c1=c1)
    else:
        raise ValueError('Method is not valid!')


    alpha_samples = np.linspace(start=0, stop=alpha_k+offset, num=4000)
    phi_alphas = [phi(func=func, xk=xk, pk=pk, alpha=alpha_i) for alpha_i in alpha_samples]

    acceptable_point = []
    not_acceptable_point = []
    if condition == 'wolfe':
        for alpha in alpha_samples:
            if wolf_conditions(func=func, func_gradient=func_gradient, xk=xk, pk=pk, alpha=alpha, c1=c1, c2=c2) == True:
                acceptable_point.append(alpha)
            else:
                not_acceptable_point.append(alpha)
    elif condition == 'golden':
        for alpha in alpha_samples:
            if goldstein(func=func, func_gradient=func_gradient, xk=xk, pk=pk, alpha=alpha, c1=c1) == True:
                acceptable_point.append(alpha)
            else:
                not_acceptable_point.append(alpha)
    else:
        raise ValueError('Condition argument isn\'t valid')

    plt.figure()
    plt.plot(alpha_samples, phi_alphas, 'b-', label='Phi(alpha)')
    plt.plot(acceptable_point, [0]*len(acceptable_point), 'r.', label='acceptable alphas are around these points')
    plt.plot(not_acceptable_point, [0] * len(not_acceptable_point), 'y.', label='not acceptable alphas are around these points')
    #plt.plot([alpha_k], [phi(func=func, xk=xk, pk=pk, alpha=alpha_k)], 'g*', label='point by backtrack')
    #plt.plot([initial_alpha], [phi(func=func, xk=xk, pk=pk, alpha=initial_alpha)], 'ro', alpha=0.3, label='alpha0')
    plt.title('phi plot - {} conditions, {}'.format(condition,description))
    plt.xlabel('alpha')
    plt.ylabel('Phi(alpha)')
    plt.legend()
