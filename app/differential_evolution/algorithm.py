# algorithm.py
import numpy as np

def sphere(x):
    return np.sum(x**2)

def rastrigin(x):

    return 10*len(x) + np.sum([xi**2 - 10*np.cos(2*np.pi*xi) for xi in x])

def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * np.array(x)))
    return -a * np.exp(-b * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + a + np.exp(1)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X, Y], axis=-1)

    funcs = [
        ('Ackley', ackley),
        ('Sphere', sphere),
        ('Rastrigin', rastrigin),
    ]

    fig, axes = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(18, 5))

    for ax, (name, func) in zip(axes, funcs):
        Z = np.apply_along_axis(func, -1, points)
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.set_title(f'{name} function')

    plt.tight_layout()
    plt.show()
    
    