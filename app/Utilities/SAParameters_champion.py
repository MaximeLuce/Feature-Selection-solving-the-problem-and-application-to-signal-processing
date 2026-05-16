from app.Problem.Problem import Problem

# OM algos
from app.OptimizationAlgorithm.EvolutionaryAlgorithm import EvolutionaryAlgorithm
from app.OptimizationAlgorithm.SimulatedAnnealing import SimulatedAnnealing
from app.OptimizationAlgorithm.RandomSearch import RandomSearch
from app.OptimizationAlgorithm.Greedy import Greedy

# ML models
from app.MLModels.HeavyModelSVM import HeavyModelSVM

import statistics
#import time
import timeit
import os
import csv

class SAParameters:
    def __init__(self):
        self.dataset_ids = [
            ## small dimension (22 features with ID=174)
            174, 
            # medium dimension (30 features with ID=17)
            17,
            # medium dimension (link signal processing) (34 features with ID=52)
            52,
            ## high dimension (link signal processing) (60 features with ID=151)
            151
        ]

        self.cases = [
            # Best initial_temp
            {"id": "1", "initial_temp": 100, "cooling_rate": 0.99},
            {"id": "2", "initial_temp": 1000, "cooling_rate": 0.99},
            {"id": "3", "initial_temp": 10000, "cooling_rate": 0.99},

            # Best cooling_rate
            #{"id": "4", "initial_temp": 1000, "cooling_rate": 0.90},
            #{"id": "5", "initial_temp": 1000, "cooling_rate": 0.99},
            #{"id": "6", "initial_temp": 1000, "cooling_rate": 0.999},
            #{"id": "7", "initial_temp": 1000, "cooling_rate": 0.9999}
        ]

    def _calc_stats(self, results):
        """Auxiliary function to computer best, worst, avg, std"""
        best = max(results)
        worst = min(results)
        avg = statistics.mean(results)
        std = statistics.stdev(results) if len(results) > 1 else 0.0
        return best, worst, avg, std

    def run_all(self):
        # main parameters
        runs_per_algo = 10 
        
        csv_filepath = "app/Results/SAParameters/SAParameters_Best_Initial_Temperature.csv"
    
        #csv_filepath = "app/Results/SAParameters/SAParameters_Best_Cooling_Rate.csv"
        
        print(f"Start running... Results will be saved to {csv_filepath}")
    

        file_exists = os.path.isfile(csv_filepath)
        with open(csv_filepath, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            
            if not file_exists:
                header = [
                    "Case_ID", "Dataset_ID", "Initial_temp", "Cooling_rate",
                    "SA_Best", "SA_Worst", "SA_Avg", "SA_Std", "SA_Time(s)", "SA_NFE_Avg",
                    "SVM_Champion_Score", "Nb_Features_Keep"
                ]
                writer.writerow(header)

            for idx, dataset_id in enumerate(self.dataset_ids):
                try:
                    problem = Problem.load_dataset(dataset_id)
                    print(f"Problem loaded: ID={dataset_id} with {problem.num_features} features and {problem.num_instances} instances.")

                    heavy_model = HeavyModelSVM(dataset_id)
                except Exception as e:
                    print(f"Error loading data: {e}")
                    continue

                
                # diplay result for the dataset_id

                print(f"\n{'='*50}")
                print(f"=== ANALYZING DATASET ID {dataset_id} ===")
                print(f"{'='*50}")
                
                for case in self.cases:
                    initial_temp = case["initial_temp"]
                    cooling_rate = case["cooling_rate"]

                    total_evals = 10

                    print(f"\n--- RUNNING CASE {case['id']} ---")
                    print(f"Params: initial_temp={initial_temp}, cooling_rate={cooling_rate}, Total Evals={total_evals}")
                    

                    # Simulated Annealing [10x]
                    sa_results_knn = []
                    sa_evals = []

                    # tracking the best overall
                    best_overall_fitness = float('inf')
                    best_overall_mask = None

                    t0_sa = timeit.default_timer()
                    for _ in range(runs_per_algo):
                        problem.reset_counter() # set to 0 the NFE counter
                        sa = SimulatedAnnealing(problem, total_evals, initial_temp, cooling_rate)
                        best_ind = sa.run()
                        sa_evals.append(problem.evaluations_count)

                        score_knn = (1.0 - best_ind.fitness) * 100
                        sa_results_knn.append(score_knn)

                        #sa_results.append(float(best_ind.fitness))
                        #sa_results.append((1.0 - best_ind.fitness) * 100)

                        if best_ind.fitness < best_overall_fitness:
                            best_overall_fitness = best_ind.fitness
                            # .copy() to not erase the reference
                            best_overall_mask = best_ind.features_mask.copy()
                            
                    t_sa = timeit.default_timer() - t0_sa
                    sa_b, sa_w, sa_a, sa_s = self._calc_stats(sa_results_knn)
                    sa_b_evals, sa_w_evals, sa_a_evals, sa_s_evals = self._calc_stats(sa_evals)

                    score_champion_heavy = heavy_model.evaluate(best_overall_mask) * 100
                    nb_features_keep = sum(best_overall_mask)

                    #print("End of computation for SimulatedAnnealing")


                    # display head of table
                    
                    
                    print("-" * 135)
                    print(f"{'Dataset_ID':<15} | {'SA (Light Model) [10x]':<26} | {'Champion SVM'}")
                    print(f"{dataset_id:<15} |  {'best*   worst  avg    std':<26} | {score_champion_heavy:.2f}% ({nb_features_keep} features)")
                    print("-" * 135)

                    sa_str = f"{sa_b:>5.0f} {sa_w:>6.0f} {sa_a:>6.1f} {sa_s:>5.1f}"
                    sa_evals_str = f"{sa_b_evals:>5.0f} {sa_w_evals:>6.0f} {sa_a_evals:>6.1f} {sa_s_evals:>5.1f}"

                    print(f"{'Score':<15} | {sa_str:<26} |")
                    print(f"{'Time':<15} |  {t_sa:>5.1f}s |")
                    print(f"{'NFE':<15} | {sa_evals_str:<26} |")
                    print("-" * 135)

                    row = [
                        case['id'], dataset_id, initial_temp, cooling_rate,
                        round(sa_b, 1), round(sa_w, 1), round(sa_a, 2), round(sa_s, 2), round(t_sa, 3), round(sa_a_evals, 1),
                        round(score_champion_heavy, 2), nb_features_keep
                    ]
                    writer.writerow(row)
                    
                    # force saving
                    file.flush()
                    print(f"Case {case['id']} saved!")