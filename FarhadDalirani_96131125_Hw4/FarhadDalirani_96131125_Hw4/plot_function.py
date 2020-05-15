# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from functions_derivatives import *
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import numpy as np


def plot_func_in_iterations_barrier(func,x_barrier_s, A, P, b, q, title='f(Xk)'):
    """
    plot f(xk)
    :return:
    """
    x=[]
    for i in range(0, len(x_barrier_s)):
        for j in range(0, len(x_barrier_s[i])):
            x.append(x_barrier_s[i][j])

    f = []
    for i in range(0, len(x)):
        f.append(func(x=x[i], A=A, P=P, b=b, q=q))

    plt.figure()
    plt.plot([i for i in range(0, len(f))], f, 'yo-')
    plt.title(title)
    plt.xlabel('Iterations(all Internal and External Iterations)')
    plt.ylabel('f(xk)')


def plot_duality_gap_barrier(x_barrier_s, duality_gaps, title='Duality Gap'):
    """
    plot Duality Gap
    :return:
    """
    dualities=[]
    for i in range(0, len(x_barrier_s)):
        for j in range(0, len(x_barrier_s[i])):
            dualities.append(duality_gaps[i])

    plt.figure()
    plt.plot([i for i in range(0, len(dualities))], dualities, 'g-')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Duality Gap')


def plot_func_in_iterations_interior(f_interior_s, title='f(Xk)'):
    """
    plot f(xk)
    :return:
    """
    plt.figure()
    plt.plot([i for i in range(0, len(f_interior_s))], f_interior_s, 'yo-')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('f(xk)')


def plot_surrogate_duality_gap_interior(surrogate_duality_gaps, title='surrogate duality gaps'):
    """
    plot surrogate duality gaps
    :return:
    """
    plt.figure()
    plt.plot([i for i in range(0, len(surrogate_duality_gaps))], surrogate_duality_gaps, 'ro-')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('surrogate duality gaps')


def plot_dual_residual_interior(r_fea_s, title='Dual Residual'):
    """
    plot Dual Residual
    :return:
    """
    plt.figure()
    plt.plot([i for i in range(0, len(r_fea_s))], r_fea_s, 'bo-')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Dual Residual')