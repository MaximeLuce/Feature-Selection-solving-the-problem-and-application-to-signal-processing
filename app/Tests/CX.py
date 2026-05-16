import random
from unittest.mock import patch
from app.Problem.Individual import Individual
from app.OptimizationAlgorithm.EvolutionaryAlgorithm import EvolutionaryAlgorithm

# TEST

if __name__ == "__main__":
    print("--- test CX ---")
    
    # example parents
    P1 = [2, 7, 4, 3, 1, 5, 6, 9, 8]
    P2 = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    expected_sequence = [2, 7, 3, 4, 1, 5, 6, 8, 9]
    
    # creating the individual from the sequence
    indP1 = Individual(P1.copy())
    indP2 = Individual(P2.copy())
    
    # creating a factice EA object
    # we use pm=1 to be sure that the mutation occurs
    ea = EvolutionaryAlgorithm(problem=None, pm=1.0, px=1.0)

    child1, child2 = ea.crossover_cx(indP1,indP2)
        
    # plot the results
    if child1.sequence == expected_sequence:
        print(f'Unit test successful, for the P1 {P1} and P2 {P2}, we get {child1.sequence} and we wanted {expected_sequence}')
    else:
        print(f'Unit test failed, for the P1 {P1} and P2 {P2}, we get {child1.sequence} instead of {expected_sequence}')