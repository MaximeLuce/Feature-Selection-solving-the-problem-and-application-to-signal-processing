from app.Problem.Problem import Problem
from app.OptimizationAlgorithm.EvolutionaryAlgorithm import EvolutionaryAlgorithm
import statistics
#import time
import timeit
import os
import csv

class EAConvergence:
    def __init__(self):
        self.instances = [
            ## easy instances
            "tai20_5_0.fsp", 
             "tai20_10_0.fsp", "tai20_20_0.fsp", 
            ## medium instances
             "tai100_10_0.fsp", "tai100_20_0.fsp",
            ## hard instances
            "tai500_20_0.fsp"
        ]
        self.best_known_value = [
            ## easy instances
             "14033",
             "20911", "33623",
            ## medium instances
             "300566", "367267",
            ## hard instances
            "6697476"
        ]

    

    def run_all(self):
        for idx, instance in enumerate(self.instances):
            problem = Problem.from_file(f"app/Data/{instance}")
            print(f"Data well loaded: {problem.num_jobs} tasks, {problem.num_machines} and matrice M = \n{problem.processing_times}.")
            
            # diplay result for the instance
            nom_inst = instance.replace(".fsp", "")
            
            # final parameters
            pop_size = 100
            generations = 1000 # most big generation size
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
            ea.logger.export_to_csv(f"app/Results/Convergence/{nom_inst}_EA_1000gen.csv")
            ea.logger.plot_save(f"Figures/convergence/{nom_inst}_EA_1000gen.png")

