import random
from unittest.mock import patch
from app.Problem.Individual import Individual
from app.OptimizationAlgorithm.EvolutionaryAlgorithm import EvolutionaryAlgorithm

# TEST

if __name__ == "__main__":
    print("--- test PMX ---")
    
    # example parents
    P1 = [7, 5, 3, 1, 9, 8, 6, 4, 2]
    P2 = [1, 3, 5, 2, 4, 6, 8, 9, 7]

    expected_sequence = [2, 3, 5, 1, 9, 8, 6, 4, 7]
    
    # creating the individual from the sequence
    indP1 = Individual(P1.copy())
    indP2 = Individual(P2.copy())
    
    # creating a factice EA object
    # we use pm=1 to be sure that the mutation occurs
    ea = EvolutionaryAlgorithm(problem=None, pm=1.0, px=1.0)
    
    # we mock the random to get surely position 4 et 6
    with patch('random.sample', return_value=[3, 5]):
        child1, child2 = ea.crossover_pmx(indP1,indP2)
        
    # plot the results
    if child2.sequence == expected_sequence:
        print(f'Unit test successful, for the P1 {P1} and P2 {P2}, we get {child1.sequence} and we wanted {expected_sequence}')
    else:
        print(f'Unit test failed, for the P1 {P1} and P2 {P2}, we get {child1.sequence} instead of {expected_sequence}')