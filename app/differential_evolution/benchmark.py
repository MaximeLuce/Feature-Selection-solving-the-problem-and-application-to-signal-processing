# benchmark.py

import numpy as np
from .algorithm import DifferentialEvolution
from .problems import AckleyProblem, RastriginProblem, SphereProblem


def plot_problem(ax, problem, config):
    x = np.linspace(problem.bounds[0, 0], problem.bounds[0, 1], 100)
    y = np.linspace(problem.bounds[1, 0], problem.bounds[1, 1], 100)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X, Y], axis=-1)

    algorithm = DifferentialEvolution(problem, config)
    result = algorithm.run()

    Z = np.apply_along_axis(problem, -1, points)
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.scatter(result.best[0], result.best[1], result.best_fitness, color='red', s=60, label='Best found')
    ax.set_title(f'{problem.name} function')
    ax.legend()

    print(f'{problem.name} best vector: {result.best}')
    print(f'{problem.name} best fitness: {result.best_fitness}')
    print(f'{problem.name} known minimum: {problem.minimum}, f(x) = {problem.minimum_value}')
    print(f'{problem.name} evaluations: {problem.evaluations_count}')


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    config = {'popsize': 50, 'max_generations': 200, 'seed': 42}

    ackley = AckleyProblem()
    sphere = SphereProblem()
    rastrigin = RastriginProblem()
    problems = [ackley, sphere, rastrigin]

    fig, axes = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(18, 5))

    for ax, problem in zip(axes, problems):
        plot_problem(ax, problem, config)

    plt.tight_layout()
    plt.show()
    
    
