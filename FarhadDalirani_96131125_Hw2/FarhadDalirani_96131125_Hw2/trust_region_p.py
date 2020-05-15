import numpy as np


def cauchy_point(gk, Bk, deltak):
    """
    Calculate pk with cauchy method
    :param gk:
    :param Bk:
    :param deltak:
    :return:
    """
    # calculate ti_k
    ti_k = 0
    temp = np.dot(np.transpose(gk), np.dot(Bk, gk))
    if temp <= 0:
        ti_k = 1
    else:
        ti_k = min(1, (np.linalg.norm(x=gk, ord=2)**3)/(deltak*temp))

    # p cauchy
    pc_k = -(ti_k)*(deltak/np.linalg.norm(x=gk, ord=2)) * gk

    return pc_k


def dogleg(gk, Bk, deltak):

    pk = None
    # Pu
    pu_numerator = -1*np.dot(np.transpose(gk), gk) * gk
    pu_denumarator = np.dot(np.transpose(gk), np.dot(Bk, gk))
    if pu_denumarator != 0:
        pu = pu_numerator/pu_denumarator
    else:
        pu = pu_numerator / (pu_denumarator+0.0000000001)

    # Pb
    pb = -(np.dot(np.linalg.inv(Bk), gk))

    ti = None
    pb_norm = np.linalg.norm(pb, ord=2)
    if pb_norm <= deltak:
        return pb


    pu_norm = np.linalg.norm(pu, ord=2)
    if pu_norm >= deltak:
        # ti is between 0 and 1
        ti = deltak/pu_norm
        pk = ti * pu
    else:
        # ti is between 1 and 2
        if np.isscalar(pu) == True or (pu.shape[0]==1 and pu.shape[1]==1):
            k = float(pu)
            z = float(pb-pu)
            t_1 = ((deltak - k) / z) + 1
            t_2 = ((-deltak - k) / z) + 1

            if t_1 >= 1 and t_1 <= 2:
                ti = t_1
            elif t_2 >= 1 and t_2 <= 2:
                ti = t_2
            else:
                raise ValueError('t_1 and t_2 are not valide!')

        else:
            k1 = pu[0, 0]
            k2 = pu[1, 0]
            temp = pb - pu
            z1 = temp[0, 0]
            z2 = temp[1, 0]
            a = z1**2 + z2**2
            b = 2*k1*z1 + 2*k2*z2
            c = k1**2 + k2**2 - deltak**2
            delta_equation = (b**2) - 4*a*c
            if delta_equation < 0:
                raise ValueError('Delta of ax^2+bx+c should be non-negative')

            t_1 = ((-b+np.sqrt(delta_equation))/(2*a)) + 1.0
            t_2 = ((-b-np.sqrt(delta_equation))/(2*a)) + 1.0

            if t_1 >= 1 and t_1 <= 2:
                ti = t_1
            elif t_2 >= 1 and t_2 <= 2:
                ti = t_2
            else:
                raise ValueError('t_1 and t_2 are not valde! t1 {}, t2 {}'.format(t_1,t_2))

        pk = pu + (ti-1)*(pb-pu)

    if pk is None:
        raise ValueError('Pk can not be none!')

    return pk


