from opdynamics.statistics.statistics import (Entropy,
                                              Proximity,
                                              Polarity,
                                              Transmissions,
                                              Acceptances,
                                              InformationDistribution)

STATISTICS = {
    "entropy": Entropy,
    "proximity": Proximity,
    "polarity": Polarity,
    "transmissions": Transmissions,
    "acceptances": Acceptances,
    "information_distribution": InformationDistribution
}