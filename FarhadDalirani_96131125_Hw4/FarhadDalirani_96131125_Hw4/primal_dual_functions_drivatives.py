import numpy as np


def f0(x, A, P, b, q):
    """
    function fn that is given in the homework 4
    :param x:
    :param A:
    :param P:
    :param b:
    :param q:
    :param t:
    :return:
    """
    if x.shape[0] != A.shape[1] or x.shape[1] != 1:
        raise ValueError('X must have dimension n*1')

    f0_value = 0.5*np.dot(np.dot(np.transpose(x), P), x) + np.dot(np.transpose(q), x)

    return float(f0_value)


def f0_gradient(x, A, P, b, q):
    """
    function fn that is given in the homework 4
    :param x:
    :param A:
    :param P:
    :param b:
    :param q:
    :param t:
    :return:
    """
    if x.shape[0] != A.shape[1] or x.shape[1] != 1:
        raise ValueError('X must have dimension n*1')

    f0_gradient_value = np.dot(P, x) + q

    return f0_gradient_value


def f0_hessian(x, A, P, b, q):
    """
    Hessian of function fn that is given in the homework 4
    :param x:
    :param A:
    :param P:
    :param b:
    :param q:
    :param t:
    :return:
    """
    if x.shape[0] != A.shape[1] or x.shape[1] != 1:
        raise ValueError('X must have dimension n*1')

    f0_hessian_value = P

    return f0_hessian_value


def f(x, A, P, b, q):
    """
    (f1,f2,...,fm)
    :param x:
    :param A:
    :param P:
    :param b:
    :param q:
    :return:
    """
    f_value = np.dot(A,x)-b
    return f_value


def f_derivative(x, A, P, b, q):
    """
    Drivative (f1,f2,...,fm)
    :param x:
    :param A:
    :param P:
    :param b:
    :param q:
    :return:
    """
    f_drivative_value = A
    return f_drivative_value


def r_dual(x, A, P, b, q, lambdas):
    """
    :param x:
    :param A:
    :param P:
    :param b:
    :param q:
    :return:
    """
    r_dual_value = f0_gradient(x=x, A=A, P=P, b=b, q=q) + np.dot(np.transpose(f_derivative(x=x, A=A, P=P, b=b, q=q)), lambdas)
    return r_dual_value


def r_cent(x, A, P, b, q, lambdas, t):
    r_cent_value = -1 * np.dot(np.diag(lambdas.T.tolist()[0]), f(x=x, A=A, P=P, b=b, q=q)) - (1/t) * np.ones((A.shape[0],1))
    return r_cent_value


def surrogate_duality_gap(x, A, P, b, q, lambdas):
    surrogate_duality_gap_value = -np.dot(np.transpose(f(x=x, A=A, P=P, b=b, q=q)), lambdas)
    return np.float(surrogate_duality_gap_value)


# test
if __name__ == '__main__':
    x = np.array([[1],[2],[3]])
    A = np.array([[2,1,1],[2,1,2],[3,3,1], [4,2,4]])
    b = np.array([[20],[20],[20], [30]])
    P = np.dot(np.array([[1], [1], [2]]), np.array([[3,1,1]]))
    q = np.array([[1],[2],[1]])

    print('>', f0(x, A, P, b, q))
    print('>>', f0_gradient(x, A, P, b, q))
    print('>>>', f0_hessian(x, A, P, b, q))

    print('+', f(x, A, P, b, q))
