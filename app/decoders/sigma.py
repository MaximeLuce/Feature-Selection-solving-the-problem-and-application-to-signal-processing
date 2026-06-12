import numpy as np


class SigmaDecoder:
    def __init__(self, bounds=None, seed=None):
        self.value_bounds = np.asarray(bounds or [-6.0, 6.0], dtype=float)
        self.name = "sigma"
        self.generator = np.random.default_rng(seed)

        if self.value_bounds.shape != (2,):
            raise ValueError("Sigma decoder bounds must be a pair: [low, high].")
        self.value_bounds = np.clip(self.value_bounds, -500, 500)

    def bounds(self, num_features: int) -> np.ndarray:
        return np.tile(self.value_bounds, (num_features, 1))

    def decode(self, vector: np.ndarray, num_features: int) -> np.ndarray:
        values = np.asarray(vector, dtype=float).reshape(-1)
        if values.size != num_features:
            raise ValueError(
                f"Sigma decoder expected {num_features} values, got {values.size}."
            )
        probabilities = 1.0 / (1.0 + np.exp(-values))
        return (self.generator.random(num_features) < probabilities).astype(int)
