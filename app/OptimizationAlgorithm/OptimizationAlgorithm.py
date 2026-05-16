from abc import ABC, abstractmethod

class OptimizationAlgorithm(ABC):
    """An abstract class from which all optimization algorithms will inherit."""
    
    def __init__(self, problem):
        self.problem = problem
        
    @abstractmethod
    def run(self):
        """Primary method that must be implemented by child classes.
        Must return an object of type ‘Individual’ (the best solution)."""
        pass