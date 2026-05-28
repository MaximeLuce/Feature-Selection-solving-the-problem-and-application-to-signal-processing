# amdecoder.py


import numpy as np
import matplotlib.pyplot as plt

def am_function(a: float, b: float, c: float, d: float, n: int) -> np.ndarray:
    x = np.linspace(0, 2, n)
    A = 2 * np.pi * c * (x - a)
    g = np.sin(2 * np.pi * (x - a) * b * np.cos(A)) + d
    return g

def decode(a: float, b: float, c: float, d: float, n: int) -> np.ndarray:
    g = am_function(a, b, c, d, n)
    arr = np.greater(g, 0)
    return arr

def plot(a: float, b: float, c: float, d: float, arr: np.ndarray) -> None:
    x = np.linspace(0, 2, 1000)
    g = am_function(a, b, c, d, len(x))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, g)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    
    if arr is not None:
        x_arr = np.linspace(0, 2, arr.shape[0])
        g_arr = am_function(a, b, c, d, arr.shape[0])
        ax.scatter(x_arr, g_arr, color='red')
        ax.set_xticks(x_arr)
        for xi, gi, vi in zip(x_arr, g_arr, arr):
            ax.annotate(f'{int(vi)}', (xi, gi), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=8, color='red')
            ax.axvline(x=xi, color='gray', linestyle=':', linewidth=0.5)
    plt.title(f'decode(a={a}, b={b}, c={c}, d={d}, n={arr.shape[0] if arr is not None else 0}) = {"".join(str(int(v)) for v in arr)}')
    plt.show()
            
if __name__ == "__main__":
    a = 0
    b = 1
    c = 1
    d = 0
    arr = decode(a, b, c, d, 11)
    plot(a, b, c, d, arr)
