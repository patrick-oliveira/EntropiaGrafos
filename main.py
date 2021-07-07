from Scripts.Model import evaluateModel
# from Scripts.Model import Model
# from Scripts.Parameters import N, pa, mu, m
# from Scripts.Model import simulate


def main():
    kappa = 10           # Parameter for information distortion tendency
    alpha = 0.5          # Proportion of individuals with tendency to increase polarity
    omega = 0.5          # Proportion of individuals with tendency to lower polarity
    lambd = 5            # Polarization coefficient
    gamma = 5.0          # Confidence factor

    seed = 42            # Seed for random algorithms initialization.

    T = 500

    statistics = {'H': [],
                  'pi': []}
    
    execution_time = evaluateModel(T, statistics, kappa, lambd, alpha, omega, gamma, seed)
    # model = Model(N, pa, mu, m, kappa, lambd, alpha, omega, gamma, seed)
    # duration = simulate(model)
    
    # print(duration)
    return execution_time

if __name__ == "__main__":
    execution_time = main()
    print(execution_time)