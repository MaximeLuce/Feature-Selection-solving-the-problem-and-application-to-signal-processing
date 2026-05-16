import random
import copy

class Individual:
    """Represents a potential solution (an individual which is a subset of features)."""
    def __init__(self, features_mask):
        self.features_mask = features_mask  # binar list
        self.fitness = float('inf') # f(x) total processing time

    # check if the mask is valid
    def check_mask(self, problem):
        # check the length of the 
        if len(self.features_mask) != problem.num_features:
            return False
        # check if the mask is well binary
        if not all(bit in [0, 1] for bit in self.features_mask):
            return False
        return True

    def evaluate(self, problem):
        """Evaluate the quality of the solution using the problem evaluation feature."""
        self.fitness = problem.evaluate(self.features_mask)
    
    #Alternative constructor: create an individual with random mask
    @classmethod
    def create_random(self, num_features):
        random_mask = [random.choice([0, 1]) for _ in range(num_features)]
        return self(random_mask)