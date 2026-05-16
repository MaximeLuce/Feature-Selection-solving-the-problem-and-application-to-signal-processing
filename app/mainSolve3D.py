import numpy as np
import matplotlib.pyplot as plt

from app.Problem.DataLoader import * 
from app.Problem.Problem import * 
from app.OptimizationAlgorithm.EvolutionaryAlgorithm import *
from app.OptimizationAlgorithm.RandomSearch import *

if __name__ == "__main__":

    filepath = "app/data/tai20_5_0.fsp"
    
    try:
        # Initizalisation
        problem = Problem.from_file(filepath)
        print(f"Data well loaded: {problem.num_jobs} tasks, {problem.num_machines} machines.")
        
        #pop_sizes = [i * 2 for i in range(10,30)]
        #generations_list = [i * 2 for i in range(10,30)]
        pop_sizes = [10, 100, 1000, 5000]
        generations_list = [10, 100, 1000, 5000]
        
        X, Y = np.meshgrid(pop_sizes, generations_list)
        Z = np.zeros(X.shape)
        
        print("Début de l'évaluation de la grille 3D...")
        
        # double loop for Z
        for i, gen in enumerate(generations_list):
            for j, p_size in enumerate(pop_sizes):
                #print(f"Test -> Pop: {p_size}, Gen: {gen}")

                ea = EvolutionaryAlgorithm(problem, pop_size=p_size, generations=gen)
                best_solution = ea.run()
                
                # we store the best result in Z
                Z[i, j] = best_solution.fitness
                
        print("Evaluations done. Generation of  the 3D graphic...")
        
        # plot
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
    
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.8)
        
        ax.set_title("Impact of Pop Size and Generations on total time")
        ax.set_xlabel('Population Size (X)')
        ax.set_ylabel('Generations (Y)')
        ax.set_zlabel('Best Fitness (Z - à minimiser)')
    
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        plt.show()

    except FileNotFoundError:
        print(f"Problem with the file {filepath} (not existing).")