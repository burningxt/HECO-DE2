from libc.math cimport cos, sin, pi, round, fabs, sqrt, log, tan
from libc.stdlib cimport rand
import random
import numpy as np


cdef int _rand_int(int r_min, int r_max):
    cdef:
        int the_int
    the_int = rand() % (r_max - r_min + 1) + r_min
    return the_int

def rand_int(r_min, r_max):
    return _rand_int(r_min, r_max)

cdef double _rand_normal(double mu, double sigma):
    cdef:
        double uniform1, uniform2, z
    uniform1 = random.random()
    uniform2 = random.random()
    z = sqrt(- 2.0 * log(uniform1)) * sin(2.0 * pi * uniform2)
    z = mu + sigma * z
    return z

def rand_normal(mu, sigma):
    return _rand_normal(mu, sigma)

cdef double _rand_cauchy(double mu, double gamma):
    cdef:
        double uniform, z
    uniform = random.random()
    z = mu + gamma * tan(pi * (uniform - 0.5))
    return z

def rand_cauchy(mu, gamma):
    return _rand_cauchy(mu, gamma)

cdef double _sgn(double v):
    cdef:
        double sgn_value
    sgn_value = -1.0
    if v > 0.0:
        sgn_value = 1.0
    elif v == 0.0:
        sgn_value = 0.0
    return sgn_value

cdef double _sum_cy(double[::1] arr):
    cdef:
        double sum_arr
        int len_arr
    len_arr = arr.shape[0]
    sum_arr = 0.0
    for i in range(len_arr):
        sum_arr += arr[i]
    return sum_arr

def sum_cy(arr):
    return _sum_cy(arr)

cdef double _np_max(double[:] z):
    cdef:
        double max_value
        int bound
    max_value = z[0]
    bound = z.shape[0]
    for i in range(bound):
        if max_value < z[i]:
            max_value = z[i]
    return max_value

def np_max(z):
    return _np_max(z)

cdef tuple _np_max_min(double[:, :] z, int dimension, int axis):
    cdef:
        double max_value, min_value
        int bound
    max_value = z[0, dimension + axis]
    min_value = z[0, dimension + axis]
    bound = z.shape[0]
    for i in range(bound):
        if max_value < z[i, dimension + axis]:
            max_value = z[i, dimension + axis]
    for i in range(bound):
        if min_value > z[i, dimension + axis]:
            min_value = z[i, dimension + axis]
    return  min_value, max_value

def np_max_min(z, dimension, axis):
    return _np_max_min(z, dimension, axis)


# cdef tuple _normalization(double[:, :]subpop_plus, double[:, ::1] weight, int dimension,  int idx):
#     cdef:
#         double equ_min, equ_max, obj_min, obj_max, vio_min, \
#             vio_max, equ_norm, obj_norm, vio_norm
#         int subpop_plus_size
#     equ_min, equ_max = _np_max_min(subpop_plus, dimension, 1)
#     obj_min, obj_max = _np_max_min(subpop_plus, dimension, 2)
#     vio_min, vio_max = _np_max_min(subpop_plus, dimension, 3)
#     dis_min, dis_max = _np_max_min(subpop_plus, dimension, 6)
#     subpop_plus_size = subpop_plus.shape[0]
#     for _ in range(subpop_plus_size):
#         if _ == 0:
#             equ_norm = (subpop_plus[_, dimension + 1] - equ_min) / (equ_max - equ_min + 1E-50)
#             subpop_plus[_, dimension] = equ_norm
#         elif _== 1:
#             vio_norm = (subpop_plus[_, dimension + 3] - vio_min) / (vio_max - vio_min + 1E-50)
#             subpop_plus[_, dimension] = vio_norm
#         else:
#             equ_norm = (subpop_plus[_, dimension + 1] - equ_min) / (equ_max - equ_min + 1E-50)
#             obj_norm = (subpop_plus[_, dimension + 2] - obj_min) / (obj_max - obj_min + 1E-50)
#             vio_norm = (subpop_plus[_, dimension + 3] - vio_min) / (vio_max - vio_min + 1E-50)
#             # dis_norm = (subpop_plus[idx, self.dimension + 6] - dis_min) / (dis_max - dis_min + 1E-50)
#             # subpop_plus[_, self.dimension + 6] = 1.0 - (1.0 - vio_norm) * dis_norm
#             subpop_plus[_, dimension] = weight[idx, 0] * equ_norm + weight[idx, 1] * vio_norm + weight[idx, 2] * obj_norm \
#                                              + weight[idx, 3] * subpop_plus[idx, dimension + 6]

cdef void _distance(double[:, ::1] pop, double[::1] optimum, int dimension):
    cdef:
        int pop_size
    pop_size = pop.shape[0]
    for i in range(pop_size):
        for j in range(dimension):
            pop[i, dimension + 6] += abs(pop[i, j] - optimum[j])

def distance(pop, optimum, dimension):
    return _distance(pop, optimum, dimension)

cdef tuple _normalization(double[:, :]subpop_plus, double[:, ::1] weight, int dimension,  int idx):
    cdef:
        double equ_min, equ_max, obj_min, obj_max, vio_min, \
            vio_max, equ_norm, obj_norm, vio_norm
        int subpop_plus_size
    equ_min, equ_max = _np_max_min(subpop_plus, dimension, 1)
    obj_min, obj_max = _np_max_min(subpop_plus, dimension, 2)
    vio_min, vio_max = _np_max_min(subpop_plus, dimension, 3)
    dis_min, dis_max = _np_max_min(subpop_plus, dimension, 6)
    subpop_plus_size = subpop_plus.shape[0]
    for _ in range(subpop_plus_size):
        equ_norm = (subpop_plus[_, dimension + 1] - equ_min) / (equ_max - equ_min + 1E-50)
        obj_norm = (subpop_plus[_, dimension + 2] - obj_min) / (obj_max - obj_min + 1E-50)
        vio_norm = (subpop_plus[_, dimension + 3] - vio_min) / (vio_max - vio_min + 1E-50)
        # dis_norm = (subpop_plus[_, dimension + 6] - dis_min) / (dis_max - dis_min + 1E-50)
        dis_norm = (dis_max - subpop_plus[_, dimension + 6]) / (dis_max - dis_min + 1E-50)
        # subpop_plus[_, dimension + 6] = 1.0 - dis_norm
        # subpop_plus[_, dimension + 6] = 1.0 - (1.0 - obj_norm) * dis_norm
        subpop_plus[_, dimension + 6] = dis_norm
        subpop_plus[_, dimension] = weight[idx, 0] * equ_norm + weight[idx, 1] * vio_norm + weight[idx, 2] * obj_norm \
                                         + weight[idx, 3] * subpop_plus[idx, dimension + 6]

def normalization(subpop_plus, weight, dimension, idx):
    return _normalization(subpop_plus, weight, dimension, idx)


cdef void _problem_18_27(int dimension, double[:]y, double[:]z):
    for i in range(dimension):
        if fabs(z[i]) < 0.5:
            y[i] = z[i]
        else:
            y[i] = 0.5 * round(2.0 * z[i])

def problem_18_27(dimension, y, z):
    _problem_18_27(dimension, y, z)


cdef tuple _problem_17_26(int dimension, double[:]y):
    cdef:
        double f, f0, f1, g1
        int i, j
    f0 = 0.0
    f1 = 1.0
    g1 = 0.0
    for i in range(dimension):
        f0 += y[i]**2
    for i in range(dimension):
        f1 *= y[i] / sqrt(1.0 + i)
        g1 += _sgn(fabs(y[i]) - (f0 - y[i]**2) - 1.0)
    f = 1.0 / 4000.0 * f0 + 1.0 - f1
    return f, g1

def problem_17_26(dimension, y):
    return _problem_17_26(dimension, y)


cdef int _rand_choice_pb_cy(int[:]arr, double[:]pb):
    cdef:
        int selected_one
        double r
    selected_one = arr[0]
    r = random.random()
    if pb[0] <= r < pb[0] + pb[1]:
        selected_one = arr[1]
    elif pb[0] + pb[1] <= r < pb[0] + pb[1] + pb[2]:
        selected_one = arr[2]
    elif pb[0] + pb[1] + pb[2] <= r <= 1.0:
        selected_one = arr[3]
    return selected_one

def rand_choice_pb_cy(arr, pb):
    return _rand_choice_pb_cy(arr, pb)


cdef tuple _rand_choice(int size):
    cdef:
        int x_1, x_2, x_3
    x_1 = _rand_int(0, size - 1)
    x_2 = _rand_int(0, size - 1)
    x_3 = _rand_int(0, size - 1)
    while x_1 == x_2:
        x_2 = _rand_int(0, size - 1)
    while x_1 == x_3 or x_2 == x_3:
        x_3 = _rand_int(0, size - 1)
    return x_1, x_2, x_3

def rand_choice(size):
    return _rand_choice(size)


def find_best(pop, axis, dimension):
    return pop[np.argmin(pop[:, dimension + axis]), :]


cdef void mutation_1(double[:, ::1]subpop, list archive, double[::1]child, double mu, double lb,
                     double ub, int idx, int lambda_, int dimension):
    cdef:
        int x_r1, x_r2, archive_size
        double[::1] best_solution_on_fitness
    archive_size = len(archive)
    x_r1 = _rand_int(0, lambda_ - 1)
    x_r2 = _rand_int(0, lambda_ + archive_size - 1)
    best_solution_on_fitness = find_best(subpop, 0, dimension)
    while x_r1 == idx:
        x_r1 = _rand_int(0, lambda_ - 1)
    while x_r2 == x_r1 or x_r2 == idx:
        x_r2 = _rand_int(0, lambda_ + archive_size - 1)
    if x_r2 < lambda_:
        for j in range(dimension):
            child[j] = subpop[idx, j] + mu * (best_solution_on_fitness[j] - subpop[idx, j]) \
                       + mu * (subpop[x_r1, j] - subpop[x_r2, j])
    else:
        for j in range(dimension):
            child[j] = subpop[idx, j] + mu * (best_solution_on_fitness[j] - subpop[idx, j]) \
                       + mu * (subpop[x_r1, j] - archive[x_r2 - lambda_][j])
    x_correction(child, dimension, lb, ub)

cdef void mutation_2(double[:, ::1]subpop, double[::1]child, double mu, double lb, double ub,
                     int lambda_, int dimension):
    cdef:
        int x_1, x_2, x_3
    x_1, x_2, x_3 = rand_choice(lambda_)
    for j in range(dimension):
        child[j] = subpop[x_1, j] + mu * (subpop[x_2, j] - subpop[x_3, j])
    x_correction(child, dimension, lb, ub)

cdef void crossover_exp(double[:, ::1]subpop, double[::1]child, int dimension, double cr, int idx):
    cdef:
        int j_rand
    j_rand = _rand_int(0, dimension - 1)
    for j in range(dimension):
        if j_rand != j and random.random() <= cr:
            child[j] = subpop[idx, j]


cdef void crossover_bi(double[:, ::1]subpop, double[::1]child, int dimension, double cr, int idx):
    cdef:
        int n, count
    n = _rand_int(0, dimension - 1)
    count = 0
    while random.random() <= cr and count < dimension:
        child[(n + count) % dimension] = subpop[idx, (n + count) % dimension]
        count += 1


cdef void _differential_evolution(double[:, ::1]subpop, list archive, double[::1]child, double mu, double cr,
                           int strategy_id, double lb, double ub, int idx, int lambda_, int dimension):
    if strategy_id == 0:
        mutation_1(subpop, archive, child, mu, lb, ub, idx, lambda_, dimension)
        crossover_exp(subpop, child, dimension, cr, idx)
    elif strategy_id == 1:
        mutation_1(subpop, archive, child, mu, lb, ub, idx, lambda_, dimension)
        crossover_bi(subpop, child, dimension, cr, idx)
    elif strategy_id == 2:
        mutation_2(subpop, child, mu, lb, ub, lambda_, dimension)
        crossover_exp(subpop, child, dimension, cr, idx)
    elif strategy_id == 3:
        mutation_2(subpop, child, mu, lb, ub, lambda_, dimension)
        crossover_bi(subpop, child, dimension, cr, idx)

def differential_evolution(subpop, archive, child, mu, cr, strategy_id, lb, ub, idx, lambda_, dimension):
    return _differential_evolution(subpop, archive, child, mu, cr, strategy_id, lb, ub, idx, lambda_, dimension)

cdef void _x_correction(double[:]child, int dimension, double lb, double ub):
    for i in range(dimension):
        if child[i] < lb:
            child[i] = min(2.0 * lb - child[i], ub)
        elif child[i] > ub:
            child[i] = max(lb, 2.0 * ub - child[i])

def x_correction(child, dimension, lb, ub):
    return _x_correction(child, dimension, lb, ub)

cdef _update_wight(double[:, ::1] pop, double[::1] median_old, double[:, ::1] weight, double[:, ::1] weight_bias,
                   int fes, int dimension, int lambda_, double tolerance, double ang):
    if fes % (100 * dimension) == 0.0:
        median_obj = np.median(pop[:, dimension + 2], axis=0)
        improvement_median_obj = (median_obj - median_old[dimension + 2]) \
                                 / (median_old[dimension + 2] + 1E-50)
        median_vio = np.median(pop[:, dimension + 3], axis=0)
        improvement_median_vio = (median_vio - median_old[dimension + 3]) \
                                 / (median_old[dimension + 3] + 1E-50)
        if median_vio > 0.0:
            if -tolerance < improvement_median_vio < tolerance:
                for i in range(lambda_):
                    weight_bias[i, 0] -= 0.1
                    weight_bias[i, 1] += 0.1
                    if weight_bias[i, 0] < 0.1:
                        weight_bias[i, 0] = 0.9
                        weight_bias[i, 1] = 1.0 - weight_bias[i, 0]
                        weight_bias[i, 3] += 0.1
                        if weight_bias[i, 3] > 1.0:
                            weight_bias[i, 3] = 0.0
                    weight[i, 0] = <double>np.random.binomial(lambda_ + 1, weight_bias[i, 0]) / lambda_
                    weight[i, 1] = <double>np.random.binomial(lambda_ + 1, weight_bias[i, 1]) / lambda_
                    weight[i, 3] = <double> np.random.binomial(lambda_ + 1, weight_bias[i, 3]) / lambda_
            elif improvement_median_vio >= 1.0 / tolerance:
                for i in range(lambda_):
                    weight_bias[i, 0] -= 0.1
                    weight_bias[i, 1] += 0.1
                    if weight_bias[i, 0] < 0.1:
                        weight_bias[i, 0] = 0.9
                        weight_bias[i, 1] = 1.0 - weight_bias[i, 0]
                    weight[i, 0] = <double>np.random.binomial(lambda_ + 1, weight_bias[i, 0]) / lambda_
                    weight[i, 1] = <double>np.random.binomial(lambda_ + 1, weight_bias[i, 1]) / lambda_
                    # weight[i, 3] = <double> np.random.binomial(lambda_ + 1, weight_bias[i, 3]) / lambda_

        elif median_vio == 0:
            if -tolerance < improvement_median_obj < tolerance:
                for i in range(lambda_):
                    weight_bias[i, 0] -= 0.1
                    weight_bias[i, 1] += 0.1
                    if weight_bias[i, 0] < 0.1:
                        weight_bias[i, 0] = 0.9
                        weight_bias[i, 1] = 1.0 - weight_bias[i, 0]
                        weight_bias[i, 3] += 0.1
                        if weight_bias[i, 3] > 1.0:
                            weight_bias[i, 3] = 0.0
                    weight[i, 0] = <double>np.random.binomial(lambda_ + 1, weight_bias[i, 0]) / lambda_
                    weight[i, 1] = <double>np.random.binomial(lambda_ + 1, weight_bias[i, 1]) / lambda_
                    weight[i, 3] = <double> np.random.binomial(lambda_ + 1, weight_bias[i, 3]) / lambda_
def update_wight(pop, median_old, weight, weight_bias, fes, dimension, lambda_, tolerance, ang):
    return _update_wight(pop, median_old, weight, weight_bias, fes, dimension, lambda_, tolerance, ang)
