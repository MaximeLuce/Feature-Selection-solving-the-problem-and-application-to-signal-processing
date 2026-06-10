import numpy as np


class SigmaDecoder:
    def __init__(self, threshold: float = 0.5, bounds=None):
        self.threshold = float(threshold)
        self.value_bounds = np.asarray(bounds or [-6.0, 6.0], dtype=float)

        if self.value_bounds.shape != (2,):
            raise ValueError("Sigma decoder bounds must be a pair: [low, high].")

    def bounds(self, num_features: int) -> np.ndarray:
        return np.tile(self.value_bounds, (num_features, 1))

    def decode(self, vector: np.ndarray, num_features: int) -> np.ndarray:
        values = np.asarray(vector, dtype=float).reshape(-1)
        if values.size != num_features:
            raise ValueError(
                f"Sigma decoder expected {num_features} values, got {values.size}."
            )

        clipped = np.clip(values, -500.0, 500.0)
        probabilities = 1.0 / (1.0 + np.exp(-clipped))
        return (probabilities >= self.threshold).astype(int)
