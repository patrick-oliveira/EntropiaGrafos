from Scripts.Model import evaluateModel
from time import time
# from Scripts.Model import Model
# from Scripts.Parameters import N, pa, mu, m
# from Scripts.Model import simulate


def main():
    kappa = 10          # Parameter for information distortion tendency
    alpha = 0.5          # Proportion of individuals with tendency to increase polarity
    omega = 0.5          # Proportion of individuals with tendency to lower polarity
    lambd = 5          # Polarization coefficient
    gamma = 5.0          # Confidence factor

    seed = int(100*(time()%1))            # Seed for random algorithms initialization.

    T = 500

    statistics = {'H': [],
                  'pi': []}
    
    evaluateModel(T, statistics, kappa, lambd, alpha, omega, gamma, seed)
    # model = Model(N, pa, mu, m, kappa, lambd, alpha, omega, gamma, seed)
    # duration = simulate(model)
    
    # print(duration)

if __name__ == "__main__":
    main()
    # a_b = [(1,6), (2,7), (3,8), (4,9), (5,10)]
    # pool = Pool()
    # result = pool.starmap(add, a_b)
    # print(result)