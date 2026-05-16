import math
import random
import copy
from app.OptimizationAlgorithm.OptimizationAlgorithm import OptimizationAlgorithm
from app.Problem.Individual import Individual

class SimulatedAnnealing(OptimizationAlgorithm):
    def __init__(self, problem, max_evaluations=10000, initial_temp=1000.0, cooling_rate=0.99):
        super().__init__(problem) # constructor of OptimizationAlgorithm
        self.max_evaluations = max_evaluations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

    def generate_neighbor(self, current_individual):
        """
        Generates a neighboring solution by applying a slight modification.
        Here, with the "bit flip" (instead of "swap" for permutation problem) method.
        """
        # create a copy of the mask to not erase the original one
        neighbor_mask = current_individual.features_mask.copy()
        
        # pic a random index to flip its value
        idx_to_flip = random.randint(0, len(neighbor_mask) - 1)
        # flip the bit
        neighbor_mask[idx_to_flip] = 1 - neighbor_mask[idx_to_flip]
        
        return Individual(neighbor_mask)

    def run(self):
        """Executes the simulated annealing algorithm."""
        
        # create a random first solution using alternative constructor of Individual
        current_sol = Individual.create_random(self.problem.num_features)
        current_sol.evaluate(self.problem)
        
        # we keep track of the best solution found
        best_sol = copy.deepcopy(current_sol)
        
        # set up temperature
        T = self.initial_temp
        
        # main loop
        for i in range(1, self.max_evaluations):
            # create and evaluate a neighbor
            neighbor = self.generate_neighbor(current_sol)
            neighbor.evaluate(self.problem)
            
            # compute the difference
            delta_f = neighbor.fitness - current_sol.fitness
            
            if delta_f < 0: # the neighbor is better
                current_sol = neighbor

                if neighbor.fitness < best_sol.fitness: # if better, we update
                    best_sol = copy.deepcopy(neighbor)
            
            # if the neighbor is worst, we accept it with a probability
            else:
                # issue : add if T > 1e-10: ? zero division ?
                probability = math.exp(-delta_f / T)
                if random.random() < probability:
                    current_sol = neighbor
                    
            # cooling down
            T = T * self.cooling_rate
            
            # warning on the zero machine
            if T < 1e-10:
                T = 1e-10
                
        return best_sol