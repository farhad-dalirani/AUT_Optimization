from functions_derivatives import *


def steepest_decent_directions(func_derivative, xk):
    """
    calculate steepest decent directions
    :param func_derivative: derivative function of function f1 or f2
    :param xk: current point it's 2*1 for f1 function and scalar for f2 function
    :return:
    """
    # calculate steepest decent direction
    negative_gradient = -1*func_derivative(x=xk)
    return negative_gradient


def quasi_newton_direction(func_derivative, func_hessian, xk, xk_plus_1, Bk=None):
    """
    Quasi-Newton direction
    :param func_derivative: derivative function of function f1 or f2
    :param func_hessian: hessian of function f1 or f2
    :param xk: previous point it's 2*1 for f1 function and scalar for f2 function
    :param xk_plus_1: current point it's 2*1 for f1 function and scalar for f2 function
    :param Bk: current approximation of Hessian
    :return:
    """
    # if k is 1, use hessian as initialization
    if Bk is None:
        Bk_plus_1 = func_hessian(xk_plus_1)
        # check eigen values are positive or not
        eigen_values = np.linalg.eig(Bk_plus_1)[0]
        min_eigen_values = np.min(eigen_values)
        if min_eigen_values <= 0:
            Bk_plus_1 = Bk_plus_1 + np.identity(Bk_plus_1.shape[0]) * (-1*min_eigen_values + 0.00001)

        pk_plus_1 = np.dot((-1 * np.linalg.inv(Bk_plus_1)), func_derivative(xk_plus_1))
        return Bk_plus_1, pk_plus_1

    Sk = xk_plus_1 - xk
    Yk = func_derivative(xk_plus_1)-func_derivative(xk)

    temp = Yk-np.dot(Bk,Sk)
    if np.dot(np.transpose(temp), Sk) != 0:
        Bk_plus_1 = Bk + (np.dot(temp, np.transpose(temp)))/(np.dot(np.transpose(temp), Sk))
    else:
        Bk_plus_1 = Bk + (np.dot(temp, np.transpose(temp))) / (np.dot(np.transpose(temp), Sk) + 0.000000001)

    # check eigen values are positive or not
    eigen_values = np.linalg.eig(Bk_plus_1)[0]
    min_eigen_values = np.min(eigen_values)
    if min_eigen_values <= 0:
        Bk_plus_1 = Bk_plus_1 + np.identity(Bk_plus_1.shape[0]) * (-1 * min_eigen_values + 0.00001)

    pk_plus_1 = np.dot((-1 * np.linalg.inv(Bk_plus_1)), func_derivative(xk_plus_1))


    return Bk_plus_1, pk_plus_1


#######################################
#           TEST part of this file
#######################################
if __name__ == '__main__':
    xk = np.array([2,3])
    xk = np.reshape(xk, newshape=[2,1])
    xk_plus_1 = xk+1
    print(steepest_decent_directions(func_derivative=f1_gradient, xk=xk))
    bk = quasi_newton_direction(func_derivative=f1_gradient, func_hessian=f1_hessian, xk=xk, xk_plus_1=xk_plus_1)
    print(bk)
    print(quasi_newton_direction(func_derivative=f1_gradient, func_hessian=f1_hessian, xk=xk, xk_plus_1=xk_plus_1, Bk=bk))
    print('===================================')
    xk = 1.000001
    xk_plus_1 = xk+1
    print(steepest_decent_directions(func_derivative=f2_gradient, xk=xk))
    bk = quasi_newton_direction(func_derivative=f2_gradient, func_hessian=f2_hessian, xk=xk, xk_plus_1=xk_plus_1)
    print(bk)
    print(quasi_newton_direction(func_derivative=f2_gradient, func_hessian=f2_hessian, xk=xk, xk_plus_1=xk_plus_1, Bk=bk))

