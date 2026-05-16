import random
from app.Problem.Individual import Individual
from app.OptimizationAlgorithm.OptimizationAlgorithm import OptimizationAlgorithm

class RandomSearch(OptimizationAlgorithm):
    def __init__(self, problem, max_evaluations=50000):
        super().__init__(problem)
        self.max_evaluations = max_evaluations

    def run(self):
        """Performs a random search and returns the best individual found."""
        best_individual = None
        
        # loop on max_evaluations
        for _ in range(self.max_evaluations):
            # random sequence generated
            seq = list(range(self.problem.num_jobs))
            random.shuffle(seq)
            
            # evaluation
            ind = Individual(seq)
            ind.evaluate(self.problem)
            
            #  update after check
            if best_individual is None or ind.fitness < best_individual.fitness:
                best_individual = ind 
                
        return best_individual