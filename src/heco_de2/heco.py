from src.benchmarks.cec2017 import Cec2017
from src.cython.slow_loops_cy import differential_evolution, rand_int, \
    rand_normal, rand_cauchy, rand_choice_pb_cy, rand_choice, \
    x_correction, normalization, sum_cy, find_best, update_wight, distance
import numpy as np
import matplotlib.pyplot as plt
from math import tan, sqrt, pi
import random


class Heco(Cec2017):
    def __init__(self, problem_id, dimension, o_shift, matrix, matrix1, matrix2):
        super().__init__(problem_id, dimension, o_shift, matrix, matrix1, matrix2)
        self.pop_size_init = 12 * self.dimension
        self.lambda_ = 20
        self.H = 5
        self.number_of_strategy = 4
        self.archive_coefficient = 1
        self.gamma = 0.1
        self.tolerance = 0.01
        self.ang = 0.05
        self.fes_max = 20000 * dimension

    def init_pop(self, pop, lb, ub):
        pop[:, :self.dimension] \
            = lb + (ub - lb) * np.random.random_sample((self.pop_size_init, self.dimension))
        for i in range(self.pop_size_init):
            self.evaluate(pop[i, :])

    def init_subpop(self, pop, subpop):
        selected_indexes = np.random.choice(pop.shape[0], self.lambda_, replace=False)
        subpop[:, :] = pop[selected_indexes, :]
        return selected_indexes

    def init_memory(self):
        memory_mu = np.full((self.number_of_strategy, self.H), 0.5)
        memory_cr = np.full((self.number_of_strategy, self.H), 0.5)
        return memory_mu, memory_cr

    def generate_mu_cr(self, memory_mu, memory_cr, success_cr, strategy_id):
        ri = rand_int(0, self.H - 1)
        cr_ri = memory_cr[strategy_id, ri]
        mu_ri = memory_mu[strategy_id, ri]
        if cr_ri == -1.0:
            cr = 0.0
        else:
            cr = rand_normal(cr_ri, 0.1)
        if cr < 0.0:
            cr = 0.0
        elif cr > 1.0:
            cr = 1.0

        mu = rand_cauchy(mu_ri, 0.1)
        while mu <= 0.0:
            mu = rand_cauchy(mu_ri, 0.1)
        if mu > 1.0:
            mu = 1.0
        return mu, cr

    @staticmethod
    def choose_strategy(strategy_ids, strategy_pb, count_success_strategy):
        sum_count = sum_cy(count_success_strategy + 2.0)
        if sum_count:
            strategy_pb = (count_success_strategy + 2.0) / sum_count
        if strategy_pb[strategy_pb < 0.05].shape[0]:
            strategy_pb[:] = 0.25
            count_success_strategy[:] = 0
        return rand_choice_pb_cy(strategy_ids, strategy_pb)

    # def distance(self, pop, optimum):
    #     pop[:, self.dimension + 6] = np.sum(abs(pop[:, :self.dimension] - optimum[:self.dimension]), axis=1)

    def eq1(self, pop):
        feasible_solutions = pop[pop[:, self.dimension + 3] == 0.0]
        fea_size = feasible_solutions.shape[0]
        if fea_size:
            f_feasible = feasible_solutions[rand_int(0, fea_size - 1), self.dimension + 2]
            pop[:, self.dimension + 1] = abs(f_feasible - pop[:, self.dimension + 2]) + f_feasible + 1E-50
        else:
            pop[:, self.dimension + 1] = pop[:, self.dimension + 2]

    def eq2(self, pop):
        feasible_solutions = pop[pop[:, self.dimension + 3] == 0.0]
        fea_size = feasible_solutions.shape[0]
        if fea_size:
            f_feasible = feasible_solutions[rand_int(0, fea_size - 1), self.dimension + 2]
            pop[:, self.dimension + 1] = pop[:, self.dimension + 2] + f_feasible + 1E-50
        else:
            pop[:, self.dimension + 1] = pop[:, self.dimension + 2]

    def eq_old(self, pop):
        best_solution_on_obj = pop[np.lexsort((pop[:, self.dimension + 2], pop[:, self.dimension + 3]))][0]
        pop[:, self.dimension + 1] = abs(pop[:, self.dimension + 2] - best_solution_on_obj[self.dimension + 2])

    def fitness_compare_1(self, pop, subpop_plus, weight, fes, idx):
        self.eq_old(subpop_plus)
        # self.distance(subpop_plus, pop[0, :])
        w_t = fes / self.fes_max
        w_i = (idx + 1) / subpop_plus.shape[0]
        weight[idx, 0] = w_t * w_i
        weight[idx, 2] = w_t * w_i + 0.1
        weight[idx, 1] = (1.0 - w_t) * (1.0 - w_i)
        weight[idx, 3] = 0.0
        normalization(subpop_plus, weight, self.dimension, idx)

    def fitness_compare_2(self, pop, subpop_plus, weight, fes, idx):
        self.eq2(subpop_plus)
        distance(subpop_plus, pop[0, :], self.dimension)
        normalization(subpop_plus, weight, self.dimension, idx)

    def selection(self, pop, subpop_plus, subpop, archive, fitness_improvements, success_mu, success_cr,
                  mu, cr, strategy_id, count_success_strategy, selected_indexes, idx):
        if subpop_plus[idx, self.dimension] > subpop_plus[-1, self.dimension]:
            success_mu[strategy_id].append(mu)
            success_cr[strategy_id].append(cr)
            fitness_improvements[strategy_id].append(abs(subpop_plus[idx, self.dimension]
                                                     - subpop_plus[-1, self.dimension]))
            archive.append(subpop_plus[idx, :].tolist())
            while len(archive) > self.archive_coefficient * pop.shape[0]:
                archive.pop(rand_int(0, len(archive) - 1))
                # for i in range(len(archive) - self.archive_coefficient * pop.shape[0]):
                #     archive.remove(archive[rand_int(0, len(archive) - 1)])

            # if not archive.shape[0]:
            #     archive = np.hstack((archive, subpop_plus[idx, :])).reshape(1, self.dimension + 10)
            # elif archive.shape[0] < self.archive_coefficient * pop.shape[0]:
            #     archive = np.vstack((archive, subpop_plus[idx, :]))
            # else:
            #     archive[rand_int(0, self.archive_coefficient * pop.shape[0] - 1), :] \
            #         = subpop_plus[idx, :]
            # while archive.shape[0] > self.archive_coefficient * pop.shape[0]:
            #     archive = np.delete(archive, rand_int(0, archive.shape[0] - 1), axis=0)

            if subpop[idx, self.dimension + 2] == pop[0, self.dimension + 2] \
                    and subpop[idx, self.dimension + 3] == pop[0, self.dimension + 3]:
                if subpop[idx, self.dimension + 3] > subpop_plus[-1, self.dimension + 3]:
                    subpop[idx, :] = subpop_plus[-1, :]
                elif subpop[idx, self.dimension + 3] == subpop_plus[-1, self.dimension + 3] \
                        and subpop[idx, self.dimension + 2] > subpop_plus[-1, self.dimension + 2]:
                    subpop[idx, :] = subpop_plus[-1, :]
            else:
                subpop[idx, :] = subpop_plus[-1, :]
            # subpop[idx, :] = subpop_plus[-1, :]
            count_success_strategy[strategy_id] += 1
        return archive

    def update_memory(self, memory_mu, memory_cr, success_mu, success_cr, fitness_improvements,
                      memory_position):
        for strategy_id in range(self.number_of_strategy):
            if len(success_mu[strategy_id]):
                arr_fitness_improvements = np.array(fitness_improvements[strategy_id])
                arr_success_mu = np.array(success_mu[strategy_id])
                arr_success_cr = np.array(success_cr[strategy_id])
                improvements_sum = np.sum(arr_fitness_improvements)
                weights = arr_fitness_improvements / (improvements_sum + 1E-50)
                mean_success_mu = sum_cy(weights * arr_success_mu**2) / (sum_cy(weights * arr_success_mu) + 1E-50)
                mean_success_cr = sum_cy(weights * arr_success_cr)
                memory_mu[strategy_id, memory_position[strategy_id]] = mean_success_mu
                memory_cr[strategy_id, memory_position[strategy_id]] = mean_success_cr
                memory_position[strategy_id] += 1
                if memory_position[strategy_id] > self.H - 1:
                    memory_position[strategy_id] = 0

    def linearly_decrease_pop_size(self, pop, fes):
        pop_size_next = 0
        if pop.shape[0] > self.lambda_:
            pop_size_next = round((self.lambda_ - self.pop_size_init) / self.fes_max * fes) \
                            + self.pop_size_init
        while self.lambda_ < pop_size_next < pop.shape[0]:
            pop = np.delete(pop, rand_int(1, pop.shape[0] - 1), 0)
            # pop = np.delete(pop, -1, 0)
        return pop

    def evolution(self):
        pop = np.zeros((self.pop_size_init, self.dimension + 10))
        optimum_old = np.zeros(self.dimension + 10)
        median_old = np.zeros(self.dimension + 10)
        subpop = np.zeros((self.lambda_, self.dimension + 10))
        child = np.zeros(self.dimension + 10)
        archive = []
        subpop_plus = np.zeros((self.lambda_ + 1, self.dimension + 10))
        # weight = np.random.binomial(self.lambda_ + 1, 0.5, (self.lambda_, 4)) / self.lambda_
        weight = np.random.random((self.lambda_, 4))
        # weight[:, 0] = np.arange(0.05, 1.05, 0.05)
        # weight[:, 1] = 1.0 - weight[:, 0]
        weight_bias = np.full((self.lambda_, 4), 0.5)
        weight[:, 2] = 0.0
        weight[:, 3] = 0.0
        memory_mu, memory_cr = self.init_memory()
        strategy_ids = np.arange(self.number_of_strategy)
        strategy_pb = np.full(self.number_of_strategy, 0.25)
        count_success_strategy = np.zeros(self.number_of_strategy)
        lb, ub = self.get_lb_ub(self.problem_id)
        self.init_pop(pop, lb, ub)
        fes = self.pop_size_init
        memory_position = [0] * self.number_of_strategy
        best_obj = pop[0, self.dimension + 2]
        best_vio = pop[0, self.dimension + 3]
        median_old[:] = np.median(pop[:, :], axis=0)

        while fes < self.fes_max:
            success_mu = [[] for _ in range(self.number_of_strategy)]
            success_cr = [[] for _ in range(self.number_of_strategy)]
            fitness_improvements = [[] for _ in range(self.number_of_strategy)]
            selected_indexes = self.init_subpop(pop, subpop)
            for idx in range(self.lambda_):
                strategy_id = self.choose_strategy(strategy_ids, strategy_pb, count_success_strategy)
                mu, cr = self.generate_mu_cr(memory_mu, memory_cr, success_cr, strategy_id)
                self.fitness_compare_2(pop, subpop, weight, fes, idx)
                differential_evolution(subpop, archive, child, mu, cr, strategy_id,
                                       lb, ub, idx, self.lambda_, self.dimension)
                self.evaluate(child)
                subpop_plus[:self.lambda_, :] = subpop
                subpop_plus[self.lambda_, :] = child
                self.fitness_compare_2(pop, subpop_plus, weight, fes, idx)
                self.selection(pop, subpop_plus, subpop, archive, fitness_improvements,
                               success_mu, success_cr, mu, cr, strategy_id, count_success_strategy,
                               selected_indexes, idx)
                fes += 1
            pop[selected_indexes, :] = subpop
            self.update_memory(memory_mu, memory_cr, success_mu, success_cr, fitness_improvements,
                               memory_position)
            update_wight(pop, median_old, weight, weight_bias, fes, self.dimension, self.lambda_, self.tolerance, self.ang)
            pop[:] = pop[np.lexsort((pop[:, self.dimension + 2], pop[:, self.dimension + 3]))]
            pop = self.linearly_decrease_pop_size(pop, fes)
            if best_vio > pop[0, self.dimension + 3]:
                best_vio = pop[0, self.dimension + 3]
                best_obj = pop[0, self.dimension + 2]
            elif best_vio == pop[0, self.dimension + 3]:
                if best_obj > pop[0, self.dimension + 2]:
                    best_obj = pop[0, self.dimension + 2]
            best_solution_on_obj = find_best(pop, 2, self.dimension)
            if fes % (100 * self.dimension) == 0.0:
                optimum_old[:] = pop[0, :]
                median_old[:] = np.median(pop[:, :], axis=0)

            # plt.clf()
            # plt.axis([np.amin(pop[:, self.dimension + 3]), np.amax(pop[:, self.dimension + 3]),
            #          np.amin(pop[:, self.dimension + 2]), np.amax(pop[:, self.dimension + 2])])
            # plt.scatter(pop[:, self.dimension + 3], pop[:, self.dimension + 2])
            # plt.xlabel("v")
            # plt.ylabel("f")
            # plt.title("Population Distribution at FES {}/{}, Weights = {}".format(fes, self.fes_max, weight),
            #           loc='center', wrap=True)
            # plt.pause(0.0001)

            # print(fes, best_obj, best_vio, np.median(pop[:, self.dimension + 2: self.dimension + 4], axis=0),
            #       best_solution_on_obj[self.dimension + 2], best_solution_on_obj[self.dimension + 3], weight[:, 3])

        plt.show()
        return best_obj, best_vio









