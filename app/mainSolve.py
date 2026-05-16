import numpy as np

from app.Problem.DataLoader import * # class to load the .fsp file
from app.Problem.Problem import * # class that contain a problem
from app.OptimizationAlgorithm.EvolutionaryAlgorithm import *
from app.OptimizationAlgorithm.RandomSearch import *

if __name__ == "__main__":

    filepath = "app/Data/tai20_10_0.fsp"
    
    try:
        # initialization of the problem
        problem = Problem.from_file(filepath)
        print(f"Data well loaded: {problem.num_jobs} tasks, {problem.num_machines} and matrice M = \n{problem.processing_times}.")
        
        # # EA solver
        # EvolutionaryAlgorithm(problem).run()

        # final parameters
        pop_size = 100
        generations = 1000
        total_evals = pop_size * generations 
        
        # for EA
        tour_size = 20
        px = 1
        pm = 1
        mutation = 'swap' # or 'inversion'
        crossover = 'pmx'
        elitism = 0

        # SA
        initial_temp = 1000
        cooling_rate = 0.999
       
        ea = EvolutionaryAlgorithm(problem, pop_size, generations-1, tour_size, px, pm, mutation, crossover, elitism)
        best_solution = ea.run()
        # export and isplay results
        ea.logger.export_to_csv(f"app/Results/Convergence/tai20_10_0_EA.csv")
        ea.logger.plot_progress()

        # # Random Search
        #pop_size = 5000
        #best_gen, best_gen_seq, worst_gen, worst_gen_seq, average = RandomSearch(problem, pop_size).run()
        #print(f"Solution for RandomSearch with {pop_size}")
        #print(f"Best solution: {best_gen} with the sequence {best_gen_seq}")
        #print(f"Worst solution: {worst_gen} with the sequence {worst_gen_seq}")
        #print(f"Average: {average}")
        
    except FileNotFoundError:
        print(f"Problem with the file {filepath} (not existing).")