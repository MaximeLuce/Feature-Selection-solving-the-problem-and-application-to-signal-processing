from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


def _am_function(a: float, b: float, c: float, d: float, n: int) -> np.ndarray:
    x = np.linspace(0, 2, n)
    phase = 2 * np.pi * c * (x - a)
    return np.sin(2 * np.pi * (x - a) * b * np.cos(phase)) + d


def _decode_am(a: float, b: float, c: float, d: float, n: int) -> np.ndarray:
    signal = _am_function(a, b, c, d, n)
    return np.greater(signal, 0).astype(int)


def plot_am(a: float, b: float, c: float, d: float, values: npt.ArrayLike) -> None:
    import matplotlib.pyplot as plt

    x = np.linspace(0, 2, 1000)
    signal = _am_function(a, b, c, d, len(x))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, signal)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)

    if values is not None:
        values = np.asarray(values)
        x_values = np.linspace(0, 2, values.shape[0])
        y_values = _am_function(a, b, c, d, values.shape[0])
        ax.scatter(x_values, y_values, color="red")
        ax.set_xticks(x_values)
        for xi, yi, value in zip(x_values, y_values, values):
            ax.annotate(
                f"{int(value)}",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                color="red",
            )
            ax.axvline(x=xi, color="gray", linestyle=":", linewidth=0.5)

    title = "".join(str(int(v)) for v in values) if values is not None else ""
    plt.title(
        f"decode_am(a={a}, b={b}, c={c}, d={d}, "
        f"n={len(values) if values is not None else 0}) = {title}"
    )
    plt.show()


class AMDecoder:
    DEFAULT_BOUNDS = {
        "a": [0.0, 2.0],
        "b": [0.0, 10.0],
        "c": [0.0, 10.0],
        "d": [-1.0, 1.0],
    }

    def __init__(self, bounds: dict | None = None):
        source_bounds = bounds or {}
        self.parameter_bounds = {
            key: list(source_bounds.get(key, default))
            for key, default in self.DEFAULT_BOUNDS.items()
        }
        self.name = "am"

    def bounds(self, num_features: int) -> np.ndarray:
        return np.asarray(
            [self.parameter_bounds[k] for k in ("a", "b", "c", "d")],
            dtype=float,
        )

    def decode_batch(
        self,
        vectors: np.ndarray,
        num_features: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=float)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != 4:
            raise ValueError(f"AM decoder expected 4 values, got {vectors.shape[1]}.")

        return np.array(
            [_decode_am(v[0], v[1], v[2], v[3], num_features) for v in vectors]
        )


class SigmaDecoder:
    def __init__(self, bounds: list | None = None):
        self.value_bounds = np.asarray(bounds or [-6.0, 6.0], dtype=float)
        self.name = "sigma"

        if self.value_bounds.shape != (2,):
            raise ValueError("Sigma decoder bounds must be a pair: [low, high].")
        self.value_bounds = np.clip(self.value_bounds, -500, 500)

    def bounds(self, num_features: int) -> np.ndarray:
        return np.tile(self.value_bounds, (num_features, 1))

    def decode_batch(
        self,
        vectors: np.ndarray,
        num_features: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=float)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != num_features:
            raise ValueError(
                f"Sigma decoder expected {num_features} values, got {vectors.shape[1]}."
            )
        return (rng.random(vectors.shape) < 1.0 / (1.0 + np.exp(-vectors))).astype(int)


DECODER_REGISTRY: dict[str, type] = {
    "sigma": SigmaDecoder,
    "am": AMDecoder,
}


def build_decoder(config: dict) -> SigmaDecoder | AMDecoder:
    if not config:
        raise ValueError("Missing decoder config.")

    name = config.get("name")
    if name not in DECODER_REGISTRY:
        valid = ", ".join(sorted(DECODER_REGISTRY))
        raise ValueError(f"Unknown decoder: {name}. Use one of: {valid}")

    return DECODER_REGISTRY[name](bounds=config.get("bounds"))
