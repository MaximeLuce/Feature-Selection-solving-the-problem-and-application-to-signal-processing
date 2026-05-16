import random
from unittest.mock import patch
from app.Problem.Individual import Individual
from app.OptimizationAlgorithm.EvolutionaryAlgorithm import EvolutionaryAlgorithm

# TEST

if __name__ == "__main__":
    print("--- test swap ---")
    
    # example sequence
    original_sequence = [2, 4, 6, 9, 1, 3, 8]
    expected_sequence = [3, 4, 6, 9, 1, 2, 8]
    
    # creating the individual from the sequence
    ind = Individual(original_sequence.copy())
    
    # creating a factice EA object
    # we use pm=1 to be sure that the mutation occurs
    ea = EvolutionaryAlgorithm(problem=None, pm=1.0)
    
    # we mock the random to get surely position 0 et 5
    with patch('random.sample', return_value=[0, 5]):
        ea.mutate_swap(ind)
        
    # plot the results
    if ind.sequence == expected_sequence:
        print(f'Unit test successful, for the sequence {original_sequence}, we get {ind.sequence} and we wanted {expected_sequence}')
    else:
        print(f'Unit test failed, for the sequence {original_sequence}, we get {ind.sequence} instead of {expected_sequence}')