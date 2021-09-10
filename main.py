from Scripts.Model import evaluateModel


def main():
    kappa = 10           # Parameter for information distortion tendency
    alpha = 0.5          # Proportion of individuals with tendency to increase polarity
    omega = 0.5          # Proportion of individuals with tendency to lower polarity
    lambd = 5            # Polarization coefficient
    gamma = 5.0          # Confidence factor

    T = 150
    
    execution_time = evaluateModel(T, kappa, lambd, alpha, omega, gamma)
    
    return execution_time

if __name__ == "__main__":
    execution_time = main()
    print(execution_time)