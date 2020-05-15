# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from functions_derivatives import *
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import numpy as np


def plot_function(func, mode, range_x1=None, range_x2=None):

    if mode == '3D':
        range_x1 = range_x1
        range_x2 = range_x2

        number_of_sample = 100

        x = np.linspace(range_x1[0], range_x1[1], number_of_sample)
        y = np.linspace(range_x2[0], range_x2[1], number_of_sample)

        z = np.ones(shape=(number_of_sample, number_of_sample))
        for i_idx, i in enumerate(x):
            for j_idx, j in enumerate(y):
                x1_x2 = [i, j]
                x1_x2 = np.reshape(x1_x2, newshape=(2,1))
                z[i_idx, j_idx] = func(x1_x2)

        x, y = np.meshgrid(x, y)



        # Set up plot
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

        ls = LightSource(270, 45)
        # To use a custom hillshading mode, override the built-in shading and pass
        # in the rgb colors of the shaded surface calculated from "shade".
        rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                               linewidth=0, antialiased=False, shade=False)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('function {}, range x1:{}, range x2:{}'.format(func.__name__, range_x1, range_x2))

    elif mode == '2D':
        range_x = range_x1
        x = np.linspace(start=range_x[0], stop=range_x[1], num=2000)
        y = [func(i) for i in x]
        plt.figure()
        plt.plot(x, y, 'b-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('function {}, range {}'.format(func.__name__, range_x))

    #plt.show()


def plot_function_with_trajectory(func, mode, range_x1=None, range_x2=None, xs=None, fs=None):

    if xs is None or fs is None:
        raise ValueError('trajectory should not be empty')

    if mode == '3D':
        x_all = np.zeros(shape=(2,len(xs)))
        for i in range(len(xs)):
            x_all[0, i] = xs[i][0, 0]
            x_all[1, i] = xs[i][1, 0]

        range_x1 = [np.min(x_all[0,:])-0.3, np.max(x_all[0,:])+0.3]
        range_x2 = [np.min(x_all[1,:])-0.3, np.max(x_all[1,:])+0.3]

        number_of_sample = 100

        x = np.linspace(range_x1[0], range_x1[1], number_of_sample)
        y = np.linspace(range_x2[0], range_x2[1], number_of_sample)

        z = np.ones(shape=(number_of_sample, number_of_sample))
        for i_idx, i in enumerate(x):
            for j_idx, j in enumerate(y):
                x1_x2 = [i, j]
                x1_x2 = np.reshape(x1_x2, newshape=(2,1))
                z[j_idx, i_idx] = func(x1_x2)

        x, y = np.meshgrid(x, y)

        # create plot
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

        ls = LightSource(270, 45)
        rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                               linewidth=0, antialiased=False, shade=False)

        for i in range(x_all.shape[1]):
            if i == 0:
                continue
            if i != x_all.shape[1]-1:
                x_i = x_all[:,i]
                x_i = np.reshape(x_i, newshape=(2,1))
                x_i_1 = x_all[:, i-1]
                x_i_1 = np.reshape(x_i_1, newshape=(2, 1))

                ax.plot([x_i_1[0,0], x_i[0,0]], [x_i_1[1,0], x_i[1,0]], [fs[i-1], fs[i]], 'b-', alpha=0.6)
            else:
                ax.plot([x_i_1[0,0], x_i[0,0]], [x_i_1[1,0], x_i[1,0]], [fs[i-1], fs[i]], 'b-', alpha=0.6, label='a step in steepest descent')
            ax.plot([x_i[0,0]], [x_i[1,0]], [fs[i]], 'y*', alpha=0.7)

        ax.plot([x_all[0,0]], [x_all[1,0]], [fs[0]], 'go', label='start')
        ax.plot([x_all[0,-1]], [x_all[1,-1]], [fs[-1]], 'mo', label='end')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('function {}, range x1:{}, range x2:{}'.format(func.__name__, range_x1, range_x2))

    elif mode == '2D':
        range_x = [np.min(xs)-0.3, np.max(xs)+0.3]
        print(range_x)
        x = np.linspace(start=range_x[0], stop=range_x[1], num=2000)
        y = [func(i) for i in x]
        plt.figure()
        plt.plot(x, y, 'r-', label='f2')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('function {}, range {}'.format(func.__name__, range_x))
        xs_p = [x_i[0,0] for x_i in xs]
        fs_p = [f_i[0, 0] for f_i in fs]
        #plt.plot(xs_p, fs_p, 'r--', label='steepest descent path')
        for i in range(len(xs_p)):
            if i == 0:
                continue
            if i != len(xs_p)-1:
                plt.plot([xs_p[i-1], xs_p[i]], [fs_p[i-1], fs_p[i]], 'b-', alpha=0.6)
            else:
                plt.plot([xs_p[i - 1], xs_p[i]], [fs_p[i - 1], fs_p[i]], 'b-', alpha=0.6, label='a step in steepest descent')
            plt.plot(xs_p[i], fs_p[i], 'y*', alpha=0.7)

        plt.plot(xs_p[0], fs_p[0], 'go', label='start')
        plt.plot(xs_p[-1], fs_p[-1], 'mo', label='end')
        plt.legend()


def plot_func_in_iterations(fs, title='f(Xk)'):
    """
    plot |f(xk+1)-f(xk)|
    :return:
    """
    fs_p = []
    for i in range(0, len(fs)):
        fs_p.append(float(fs[i]))

    plt.figure()
    plt.plot([i for i in range(0, len(fs_p))], fs_p, 'yo-')
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('f(xk)')


def plot_decrease_in_iterations(fs, title='|f(xk+1)-f(xk)|'):
    """
    plot |f(xk+1)-f(xk)|
    :return:
    """
    fs_difference = []
    for i in range(1, len(fs)):
        fs_difference.append(abs(float(fs[i])-float(fs[i-1])))

    plt.figure()
    plt.plot([i for i in range(0, len(fs_difference))], fs_difference, 'go-')
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('|f(xk+1)-f(xk)|')


def plot_ru_in_iterations(ru_s, title='ru in iterations'):
    """
    plot |f(xk+1)-f(xk)|
    :return:
    """
    ru_s_p = []
    for i in range(0, len(ru_s)):
        ru_s_p.append(float(ru_s[i]))

    plt.figure()
    plt.plot([i for i in range(0, len(ru_s_p))], ru_s_p, 'yo-')
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('RUk')


def plot_norm_pk_in_iterations(pk_s, title='norm pk in iterations'):
    """
    plot norm pk
    :return:
    """
    pk_s_p = []
    for i in range(0, len(pk_s)):
        pk_s_p.append(np.linalg.norm(pk_s[i], ord=2))

    plt.figure()
    plt.plot([i for i in range(0, len(pk_s_p))], pk_s_p, 'yo-')
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Norm Pk')


#######################################
#           TEST part of this file
#######################################
if __name__ == "__main__":
    # plot f1 in to different range
    plot_function(func=f1, mode='3D', range_x1=[-2, 2], range_x2=[-1,3])
    plot_function(func=f1, mode='3D', range_x1=[-5, 5], range_x2=[-5, 5])

    # plot f2
    plot_function(func=f2, mode='2D', range_x1=[-2, 3])
    plot_function(func=f2, mode='2D', range_x1=[-0.1, 1.2])

    plt.show()