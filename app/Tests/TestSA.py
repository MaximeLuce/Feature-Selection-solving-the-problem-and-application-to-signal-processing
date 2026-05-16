import time
import sys
import os

from app.Problem.Problem import Problem
from app.OptimizationAlgorithm.SimulatedAnnealing import SimulatedAnnealing

def run_test():
    print("Starting test with SA...")

    print("Loading specific dataset id=151...")
    try:
        problem = Problem.load_dataset(151)
        print(problem)
    except Exception as e:
        print(f"Error loading the dataset: {e}")
        return

    # fake parameters for SA
    max_evals = 100
    initial_temp = 10.0
    cooling_rate = 0.95
    
    print("Ruuning Simulated Annealing...")
    sa = SimulatedAnnealing(problem, max_evals, initial_temp, cooling_rate)
    print(f"Parameters: {max_evals} iterations, T0={initial_temp}, cooling_rate={cooling_rate}")
    start_time = time.time()
    best_solution = sa.run()
    end_time = time.time()

    print("Results...")
    
    # the accuracy is 1-error (fitness)
    best_accuracy = (1.0 - best_solution.fitness) * 100
    
    # indexes of selected features
    selected_features = [i for i, bit in enumerate(best_solution.features_mask) if bit == 1]
    
    print(f"Execution time: {end_time - start_time:.2f} secondes")
    print(f"Accuracy: {best_accuracy:.2f}% (Eroor of {best_solution.fitness*100:.2f}%)")
    print(f"Keep features: {len(selected_features)} over {problem.num_features}")
    print(f"Indexes of features: {selected_features}")

if __name__ == "__main__":
    run_test()