import numpy as np


class SigmaDecoder:
    name = "sigma"

    def __init__(self, bounds=None):
        self.lower, self.upper = tuple(bounds or (-6.0, 6.0))

    def bounds(self, num_features):
        return np.tile([self.lower, self.upper], (num_features, 1))

    def decode_batch(self, vectors, num_features, rng):
        vectors = np.asarray(vectors, dtype=float)
        probabilities = 1.0 / (1.0 + np.exp(-vectors))
        return (rng.random((len(vectors), num_features)) < probabilities).astype(int)


class AMDecoder:
    name = "am"

    def __init__(self, bounds=None):
        self.param_bounds = bounds or {
            "a": (0.0, 2.0),
            "b": (0.0, 10.0),
            "c": (0.0, 10.0),
            "d": (-1.0, 1.0),
        }

    def bounds(self, num_features):
        return np.array(
            [
                self.param_bounds["a"],
                self.param_bounds["b"],
                self.param_bounds["c"],
                self.param_bounds["d"],
            ],
            dtype=float,
        )

    def decode_batch(self, vectors, num_features, rng):
        vectors = np.asarray(vectors, dtype=float)
        x = np.linspace(0.0, 2.0 * np.pi, num_features, endpoint=False)
        masks = []
        for a, b, c, d in vectors:
            signal = a * np.sin(b * x + d) + np.cos(c * x - d)
            probabilities = 1.0 / (1.0 + np.exp(-signal))
            masks.append((rng.random(num_features) < probabilities).astype(int))
        return np.asarray(masks, dtype=int)


def build_decoder(config):
    name = config["name"].lower()
    if name == "sigma":
        return SigmaDecoder(config.get("bounds"))
    if name == "am":
        return AMDecoder(config.get("bounds"))
    raise ValueError(f"Unknown decoder: {config['name']}")
