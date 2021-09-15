from Scripts.Model import evaluateModel, initialize_model
from Scripts.Parameters import N, memory_size, code_length, kappa, alpha, omega, lambd, gamma, prefferential_att, \
                               num_repetitions, T, seed


def main():
    model = initialize_model(N, prefferential_att, memory_size, code_length, kappa, lambd, alpha, omega, gamma, seed)
    elapsedTime, rep_statistics, mean_statistics = evaluateModel(model, T, num_repetitions = num_repetitions)
    
    return elapsedTime, rep_statistics, mean_statistics

if __name__ == "__main__":
    elapsedTime, rep_statistics, mean_statistics = main()
    print(elapsedTime)