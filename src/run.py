from src.heco_de1.heco import Heco
from src.benchmarks.cec2017 import load_mat
import timeit
import multiprocessing as mp


def run(runs):
    # for dimension in [10, 30, 50, 100]:
    for dimension in [100]:
        # for problem_id in [3, 4, 6, 8, 9, 10, 18, 25]:#range(1, 29):
        # for problem_id in [1, 2, 5, 7, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 26, 27, 28]:  # range(1, 29):
        for problem_id in range(1, 29):
            o, m, m1, m2 = load_mat(problem_id, dimension)
            results = Heco(problem_id, dimension, o, m, m1, m2).evolution()
            print(results[0], file=open("./data_analysis/output_data/F{}_{}D_obj.txt".format(problem_id, dimension), "a"))
            print(results[1], file=open("./data_analysis/output_data/F{}_{}D_vio.txt".format(problem_id, dimension), "a"))


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    start = timeit.default_timer()
    pool = mp.Pool(processes=25)
    res = pool.map(run, range(25))
    stop = timeit.default_timer()
    print('Time: ', stop - start)


# if __name__ == '__main__':
#     start = timeit.default_timer()
#     prob_id = 6
#     dim = 100
#     o, m, m1, m2 = load_mat(prob_id, dim)
#     Heco(prob_id, dim, o, m, m1, m2).evolution()
#     stop = timeit.default_timer()
#     print('Time: ', stop - start)


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
            #       best_solution_on_obj[self.dimension + 2], best_solution_on_obj[self.dimension + 3],
            #       weight[0], weight[3])

