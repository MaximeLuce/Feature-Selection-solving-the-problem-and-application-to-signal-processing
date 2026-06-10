# fitness.py

class Fitness:
    name = "fitness"
    
    def compute(self, error, feature_ratio) -> float:
        raise NotImplementedError

class ErrorFitness(Fitness):
    name = "error"

    def compute(self, error, feature_ratio : float = 0.0):
        return error


class WeightedErrorFitness(Fitness):
    name = "weighted_error"

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def compute(self, error: float, feature_ratio: float):
        return self.alpha * error + ((1.0 - self.alpha) * feature_ratio)

