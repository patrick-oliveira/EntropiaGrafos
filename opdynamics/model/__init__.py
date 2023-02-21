from .model import Model
from .statistics import (InformationDistribution, MeanAcceptances, MeanEntropy,
                         MeanPolarity, MeanProximity, MeanTransmissions)

stats_dict = {
    "Entropy": MeanEntropy,
    "Proximity": MeanProximity,
    "Polarity": MeanPolarity,
    "Distribution": InformationDistribution,
    "Acceptance": MeanAcceptances,
    "Transmission": MeanTransmissions
}