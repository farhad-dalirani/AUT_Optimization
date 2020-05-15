from functions_derivatives import *


def newton_decent_directions(function, func_derivative, func_hessian, xk, A, P, b, q, t):
    """
    calculate steepest decent directions
    :param func_derivative: derivative function of function f1 or f2
    :param xk: current point it's 2*1 for f1 function and scalar for f2 function
    :return:
    """
    # calculate steepest decent direction
    newton_dir = -np.dot(np.linalg.inv(func_hessian(x=xk, A=A, P=P, b=b, q=q, t=t)), func_derivative(x=xk, A=A, P=P, b=b, q=q, t=t))

    return newton_dir

