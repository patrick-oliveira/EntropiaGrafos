from Scripts.ModelDynamics import evaluateModel


def main():
    kappa = 10          # Parameter for information distortion tendency
    alpha = 0.5          # Proportion of individuals with tendency to increase polarity
    omega = 0.5          # Proportion of individuals with tendency to lower polarity
    lambd = 5          # Polarization coefficient
    gamma = 5.0          # Confidence factor

    seed = int(100*(time.time()%1))            # Seed for random algorithms initialization.

    T = 15000

    statistics = {'H': [],
                  'pi': []}
    
#     evaluateModel(T, statistics, kappa, lambd, alpha, omega, gamma, seed)
    model = Model(N, pa, mu, m, kappa, lambd, alpha, omega, gamma, seed)
    duration = simulate(model)

if __name__ == "__main__":
    main()