import numpy as np


def am_function(a: float, b: float, c: float, d: float, n: int) -> np.ndarray:
    x = np.linspace(0, 2, n)
    phase = 2 * np.pi * c * (x - a)
    return np.sin(2 * np.pi * (x - a) * b * np.cos(phase)) + d


def decode_am(a: float, b: float, c: float, d: float, n: int) -> np.ndarray:
    signal = am_function(a, b, c, d, n)
    return np.greater(signal, 0).astype(int)


def plot_am(a: float, b: float, c: float, d: float, values: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    x = np.linspace(0, 2, 1000)
    signal = am_function(a, b, c, d, len(x))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, signal)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)

    if values is not None:
        x_values = np.linspace(0, 2, values.shape[0])
        y_values = am_function(a, b, c, d, values.shape[0])
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

    title = "".join(str(int(value)) for value in values) if values is not None else ""
    plt.title(
        f"decode_am(a={a}, b={b}, c={c}, d={d}, n={len(values) if values is not None else 0}) = {title}"
    )
    plt.show()


class AMDecoder:
    DEFAULT_BOUNDS = {
        "a": [0.0, 2.0],
        "b": [0.0, 10.0],
        "c": [0.0, 10.0],
        "d": [-1.0, 1.0],
    }

    def __init__(self, bounds=None):
        source_bounds = bounds or {}
        self.parameter_bounds = {
            key: list(source_bounds.get(key, default))
            for key, default in self.DEFAULT_BOUNDS.items()
        }

    def bounds(self, num_features: int) -> np.ndarray:
        return np.asarray(
            [
                self.parameter_bounds["a"],
                self.parameter_bounds["b"],
                self.parameter_bounds["c"],
                self.parameter_bounds["d"],
            ],
            dtype=float,
        )

    def decode(self, vector: np.ndarray, num_features: int) -> np.ndarray:
        values = np.asarray(vector, dtype=float).reshape(-1)
        if values.size != 4:
            raise ValueError(f"AM decoder expected 4 values, got {values.size}.")

        return decode_am(values[0], values[1], values[2], values[3], num_features)
