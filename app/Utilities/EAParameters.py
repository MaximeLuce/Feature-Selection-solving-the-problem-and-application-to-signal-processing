from app.Problem.Problem import Problem
from app.OptimizationAlgorithm.EvolutionaryAlgorithm import EvolutionaryAlgorithm
import statistics
#import time
import timeit
import os
import csv

class EAParameters:
    def __init__(self):
        self.instances = [
            ## easy instances
            #"tai20_5_0.fsp", 
            # "tai20_10_0.fsp", "tai20_20_0.fsp", 
            ## medium instances
            # "tai100_10_0.fsp", "tai100_20_0.fsp",
            ## hard instances
            "tai500_20_0.fsp"
        ]
        self.best_known_value = [
            ## easy instances
            # "14033",
            # "20911", "33623",
            ## medium instances
            # "300566", "367267",
            ## hard instances
            "6697476"
        ]

        self.cases = [
            # TABLE 1: Best Ratio Population/Generations Size
            #{"id": 1, "pop_size": 100, "generations": 100, "px": 0.7, "pm": 0.1, "tour": 5, "mutation": "swap", "crossover": "ox", "elitism": 0},
            #{"id": 2, "pop_size": 20, "generations": 500, "px": 0.7, "pm": 0.1, "tour": 5, "mutation": "swap", "crossover": "ox", "elitism": 0},
            #{"id": 3, "pop_size": 1000, "generations": 10, "px": 0.7, "pm": 0.1, "tour": 5, "mutation": "swap", "crossover": "ox", "elitism": 0},
            #{"id": 4, "pop_size": 10000, "generations": 1, "px": 0.7, "pm": 0.1, "tour": 5, "mutation": "swap", "crossover": "ox", "elitism": 0}
            
            # TABLE 2: Best Crossover Probability
            #{"id": 5, "pop_size": 100, "generations": 100, "px": 0.0, "pm": 0.1, "tour": 5, "mutation": "swap", "crossover": "ox", "elitism": 0},
            #{"id": 6, "pop_size": 100, "generations": 100, "px": 0.3, "pm": 0.1, "tour": 5, "mutation": "swap", "crossover": "ox", "elitism": 0},
            #{"id": 7, "pop_size": 100, "generations": 100, "px": 0.7, "pm": 0.1, "tour": 5, "mutation": "swap", "crossover": "ox", "elitism": 0},
            #{"id": 8, "pop_size": 100, "generations": 100, "px": 1.0, "pm": 0.1, "tour": 5, "mutation": "swap", "crossover": "ox", "elitism": 0}
            
            # TABLE 3: Best Mutation Probability
            #{"id": 9, "pop_size": 100, "generations": 100, "px": 1, "pm": 0.0, "tour": 5, "mutation": "swap", "crossover": "ox", "elitism": 0},
            #{"id": 10, "pop_size": 100, "generations": 100, "px": 1, "pm": 0.01, "tour": 5, "mutation": "swap", "crossover": "ox", "elitism": 0},
            #{"id": 11, "pop_size": 100, "generations": 100, "px": 1, "pm": 0.1, "tour": 5, "mutation": "swap", "crossover": "ox", "elitism": 0},
            #{"id": 12, "pop_size": 100, "generations": 100, "px": 1, "pm": 1.0, "tour": 5, "mutation": "swap", "crossover": "ox", "elitism": 0}
            
            # TABLE 4: Best Tour Size
            #{"id": 13, "pop_size": 100, "generations": 100, "px": 1, "pm": 1, "tour": 1, "mutation": "swap", "crossover": "ox", "elitism": 0},
            #{"id": 14, "pop_size": 100, "generations": 100, "px": 1, "pm": 1, "tour": 5, "mutation": "swap", "crossover": "ox", "elitism": 0},
            #{"id": 15, "pop_size": 100, "generations": 100, "px": 1, "pm": 1, "tour": 20, "mutation": "swap", "crossover": "ox", "elitism": 0},
            #{"id": 16, "pop_size": 100, "generations": 100, "px": 1, "pm": 1, "tour": 100, "mutation": "swap", "crossover": "ox", "elitism": 0}
            
            # TABLE 5: Best Mutation Type
            #{"id": 17, "pop_size": 100, "generations": 100, "px": 1, "pm": 1, "tour": 20, "mutation": "swap", "crossover": "ox", "elitism": 0},
            #{"id": 18, "pop_size": 100, "generations": 100, "px": 1, "pm": 1, "tour": 20, "mutation": "inversion", "crossover": "ox", "elitism": 0}
            
            # TABLE 6: Best Crossover Type
            #{"id": 19, "pop_size": 100, "generations": 100, "px": 1, "pm": 1, "tour": 20, "mutation": "swap", "crossover": "ox", "elitism": 0},
            #{"id": 20, "pop_size": 100, "generations": 100, "px": 1, "pm": 1, "tour": 20, "mutation": "swap", "crossover": "pmx", "elitism": 0},
            #{"id": 21, "pop_size": 100, "generations": 100, "px": 1, "pm": 1, "tour": 20, "mutation": "swap", "crossover": "cx", "elitism": 0}
            
            # TABLE 7: Impact Elitism
            {"id": 22, "pop_size": 100, "generations": 100, "px": 1, "pm": 1, "tour": 20, "mutation": "swap", "crossover": "pmx", "elitism": 0},
            {"id": 23, "pop_size": 100, "generations": 100, "px": 1, "pm": 1, "tour": 20, "mutation": "swap", "crossover": "pmx", "elitism": 1},
            {"id": 24, "pop_size": 100, "generations": 100, "px": 1, "pm": 1, "tour": 20, "mutation": "swap", "crossover": "pmx", "elitism": 5}
            
            ]
            
        

    def _calc_stats(self, results):
        """Auxiliary function to computer best, worst, avg, std"""
        best = min(results)
        worst = max(results)
        avg = statistics.mean(results)
        std = statistics.stdev(results) if len(results) > 1 else 0.0
        return best, worst, avg, std

    def run_all(self):
        # main parameters
        runs_per_algo = 10

        """TABLE 1"""
        #csv_filepath = "app/Results/EAParameters/EAParameters_Best_Ratio_Generation_Population_Basic_Instances.csv"
        #csv_filepath = "app/Results/EAParameters/EAParameters_Best_Ratio_Generation_Population_Medium_Instances.csv"
        #csv_filepath = "app/Results/EAParameters/EAParameters_Best_Ratio_Generation_Population_Hard_Instance.csv"

        """TABLE 2"""
        #csv_filepath = "app/Results/EAParameters/EAParameters_Best_Crossover_Probability_Basic_Instances.csv"
        #csv_filepath = "app/Results/EAParameters/EAParameters_Best_Crossover_Probability_Medium_Instances.csv"
        #csv_filepath = "app/Results/EAParameters/EAParameters_Best_Crossover_Probability_Hard_Instance.csv"
        

        """TABLE 3"""
        #csv_filepath = "app/Results/EAParameters/EAParameters_Best_Mutation_Probability_Basic_Instances.csv"
        #csv_filepath = "app/Results/EAParameters/EAParameters_Best_Mutation_Probability_Medium_Instances.csv"
        #csv_filepath = "app/Results/EAParameters/EAParameters_Best_Mutation_Probability_Hard_Instances.csv"
        

        """TABLE 4"""
        #csv_filepath = "app/Results/EAParameters/EAParameters_Best_Tour_Size_Basic_Instances.csv"
        #csv_filepath = "app/Results/EAParameters/EAParameters_Best_Tour_Size_Medium_Instances.csv"
        #csv_filepath = "app/Results/EAParameters/EAParameters_Best_Tour_Size_Hard_Instance.csv"
        

        """TABLE 5"""
        #csv_filepath = "app/Results/EAParameters/EAParameters_Best_Mutation_Type_Basic_Instances.csv"
        #csv_filepath = "app/Results/EAParameters/EAParameters_Best_Mutation_Type_Medium_Instances.csv"
        #csv_filepath = "app/Results/EAParameters/EAParameters_Best_Mutation_Type_Hard_Instance.csv"
        

        """TABLE 6"""
        #csv_filepath = "app/Results/EAParameters/EAParameters_Best_Crossover_Type_Basic_Instances.csv"
        #csv_filepath = "app/Results/EAParameters/EAParameters_Best_Crossover_Type_Medium_Instances.csv"
        #csv_filepath = "app/Results/EAParameters/EAParameters_Best_Crossover_Type_Hard_Instance.csv"
        

        """TABLE 7"""
        #csv_filepath = "app/Results/EAParameters/EAParameters_Elitism_Basic_Instances.csv"
        #csv_filepath = "app/Results/EAParameters/EAParameters_Elitism_Medium_Instances.csv"
        csv_filepath = "app/Results/EAParameters/EAParameters_Elitism_Hard_Instance.csv"
        
        
        print(f"Start running... Results will be saved to {csv_filepath}")
        
        file_exists = os.path.isfile(csv_filepath)
        with open(csv_filepath, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            
            if not file_exists:
                header = [
                    "Case_ID", "Instance", "Population Size", "Generations number", "Px", "Pm", "Tour", "Mutation Type", "Crossover Type",
                    "EA_Best", "EA_Worst", "EA_Avg", "EA_Std", "EA_Time(s)", "EA_NFE_Avg",
                    "Best_Known_Value"
                ]
                writer.writerow(header)

            for idx, instance in enumerate(self.instances):
                problem = Problem.from_file(f"app/Data/{instance}")
                best_known_value = self.best_known_value[idx]

                
                # diplay result for the instance
                nom_inst = instance.replace(".fsp", "")

                print(f"\n{'='*50}")
                print(f"=== ANALYZING INSTANCE: {nom_inst} ===")
                print(f"{'='*50}")
                
                for case in self.cases:
                    print(f"\n--- RUNNING CASE {case['id']} ---")
                    print(f"Params: Pop={case['pop_size']}, Gen={case['generations']}, Total Evals={case['pop_size'] * case['generations']}")
                    
                    pop_size = case["pop_size"]
                    generations = case["generations"]
                    tour_size = case["tour"]
                    px = case["px"]
                    pm = case["pm"]
                    mutation = case["mutation"]
                    crossover = case["crossover"]
                    elitism = case["elitism"]
                    
                    total_evals = pop_size * generations


                    # Evolutionary Algorithm [10x]
                    ea_results = []
                    ea_evals = []
                    t0_ea = timeit.default_timer()
                    for _ in range(runs_per_algo):
                        problem.reset_counter() # set to 0 the NFE counter
                        ea = EvolutionaryAlgorithm(problem, pop_size, generations-1, tour_size, px, pm, mutation, crossover, elitism)
                        best_ind = ea.run()
                        ea_results.append(float(best_ind.fitness))
                        ea_evals.append(problem.evaluations_count)
                    t_ea = timeit.default_timer() - t0_ea
                    ea_b, ea_w, ea_a, ea_s = self._calc_stats(ea_results)
                    ea_b_evals, ea_w_evals, ea_a_evals, ea_s_evals = self._calc_stats(ea_evals)

                    print("End of computation for EA")

                    # display head of table
                    
                    
                    print("-" * 135)
                    print(f"{'Instance':<15} | {'Evol. Alg. [10x]':<26} | {'Best known value'}")
                    print(f"{nom_inst:<15} |  {'best*   worst  avg    std':<26}  |")
                    print("-" * 135)

                    ea_str = f"{ea_b:>5.0f} {ea_w:>6.0f} {ea_a:>6.1f} {ea_s:>5.1f}"
                   
                    ea_evals_str = f"{ea_b_evals:>5.0f} {ea_w_evals:>6.0f} {ea_a_evals:>6.1f} {ea_s_evals:>5.1f}"
                    
                    print(f"{'Score':<15} | {ea_str:<26} | {best_known_value}")
                    print(f"{'Time':<15} | {t_ea:>7.2f}s {'':<17} |")
                    print(f"{'NFE':<15} | {ea_evals_str:<26} |")
                    print("-" * 135)

                    row = [
                        case['id'], instance, pop_size, generations, px, pm, tour_size, mutation, crossover,
                        round(ea_b, 1), round(ea_w, 1), round(ea_a, 2), round(ea_s, 2), round(t_ea, 3), round(ea_a_evals, 1),                        best_known_value
                    ]
                    writer.writerow(row)
                    
                    # force saving
                    file.flush()
                    print(f"Case {case['id']} saved!")