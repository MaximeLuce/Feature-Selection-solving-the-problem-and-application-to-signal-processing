import random
from unittest.mock import patch
from app.Problem.Individual import Individual
from app.OptimizationAlgorithm.EvolutionaryAlgorithm import EvolutionaryAlgorithm

if __name__ == "__main__":
    print("--- Test Tournament ---")
    
    # fake population
    population = []
    fitnesses = [10, 20, 30, 40, 50]
    
    for fit in fitnesses:
        ind = Individual([1, 2, 3]) 
        ind.fitness = fit 
        population.append(ind)
        
    # creatung the ea
    ea = EvolutionaryAlgorithm(problem=None, tour_size=3)
    ea.population = population
    
    # the tournament will select individual 2, 3 and 4 (with fitness 30, 40 and 50)
    participants_mock = [population[2], population[3], population[4]]
    
    # we moke the random
    with patch('random.sample', return_value=participants_mock):
        selected_parent = ea.tournament_selection()
        
    # test : best of three ?
    expected_best_ind = population[2]
    logic_passed = selected_parent.fitness == expected_best_ind.fitness
    
    # test : a real copy and not in memory ?
    deepcopy_passed = id(selected_parent) != id(expected_best_ind)
    
    # display
    if logic_passed and deepcopy_passed:
        print("Unit test successful!")
        print(f"Participants fitnesses : {[p.fitness for p in participants_mock]}")
        print(f"Selected fitness : {selected_parent.fitness} (Expected: 30)")
        print(f"Memory ID of original : {id(expected_best_ind)}")
        print(f"Memory ID of selected : {id(selected_parent)}")
    else:
        print("Unit test failed!")
        if not logic_passed:
            print(f"Logic Error : Expected fitness 30 but got {selected_parent.fitness}")
        if not deepcopy_passed:
            print("Memory Error : The selected parent is a ref of the original, not a deepcopy")