from app.Problem.Problem import Problem
from app.OptimizationAlgorithm.EvolutionaryAlgorithm import EvolutionaryAlgorithm
from app.OptimizationAlgorithm.SimulatedAnnealing import SimulatedAnnealing
from app.OptimizationAlgorithm.RandomSearch import RandomSearch
from app.OptimizationAlgorithm.Greedy import Greedy
import statistics
import time
import os
import csv

class ComparisonRunner:
    def __init__(self):
        self.instances = [
            ## easy instances
            "tai20_5_0.fsp", 
             "tai20_10_0.fsp", "tai20_20_0.fsp"
            ## medium instances
            # "tai100_10_0.fsp", "tai100_20_0.fsp"
            ## hard instances
            #"tai500_20_0.fsp"
        ]
        self.best_known_value = [
            ## easy instances
             "14033",
             "20911", "33623"
            ## medium instances
            # "300566", "367267"
            ## hard instances
            #"6697476"
        ]


    def _calc_stats(self, results):
        """Auxiliary function to computer best, worst, avg, std"""
        best = min(results)
        worst = max(results)
        avg = statistics.mean(results)
        std = statistics.stdev(results) if len(results) > 1 else 0.0
        return best, worst, avg, std
    
    def _find_best_sequence(self, sequences, results):
        best = int(min(results))
        index = results.index(best)
        return sequences[index]

    def run_all(self):
        # main parameters
        runs_per_algo = 10 
        
        csv_filepath = "app/Results/Final_Comparison_Final_Parameters_Basic_Instances_best_sequence.csv"
        #csv_filepath = "app/Results/Final_Comparison_Final_Parameters_Medium_Instances_best_sequence.csv"
        #csv_filepath = "app/Results/Final_Comparison_Final_Parameters_Hard_Instance_best_sequence.csv"
        
        print(f"Start running... Results will be saved to {csv_filepath}")
        
        # final parameters
        pop_size = 100
        generations = 100
        total_evals = pop_size * generations 
        
        # for EA
        tour_size = 20
        px = 1
        pm = 1
        mutation = 'swap' # or 'inversion'
        crossover = 'pmx'
        elitism = 0

        # SA
        initial_temp = 1000
        cooling_rate = 0.999

        file_exists = os.path.isfile(csv_filepath)
        with open(csv_filepath, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            
            if not file_exists:
                header = [
                    "Instance", "pop_size", "generations", "Px", "Pm", "Tour", "Mutation_Type", "Crossover_Type", "elitism", "initial_temp", "cooling_rate",
                    "RS_Best", "RS_Worst", "RS_Avg", "RS_Std", "RS_Time(s)", "RS_NFE", "RS_Best_Sequence",
                    "GR_Best", "GR_Time(s)", "GR_NFE", "GR_Best_Sequence",
                    "EA_Best", "EA_Worst", "EA_Avg", "EA_Std", "EA_Time(s)", "EA_NFE_Avg", "EA_Best_Sequence",
                    "SA_Best", "SA_Worst", "SA_Avg", "SA_Std", "SA_Time(s)", "SA_NFE_Avg", "SA_Best_Sequence",
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
                
                
                total_evals = pop_size * generations

                
                # Random Search [10x]
                rs_results = []
                rs_evals = []
                rs_sequences = []
                t0_rs = time.perf_counter()
                for _ in range(runs_per_algo):
                    problem.reset_counter() # set to 0 the NFE counter
                    rs = RandomSearch(problem, total_evals)
                    best_ind = rs.run()
                    rs_results.append(float(best_ind.fitness))
                    rs_sequences.append(best_ind.sequence)
                    rs_evals.append(problem.evaluations_count)
                t_rs = time.perf_counter() - t0_rs
                rs_b, rs_w, rs_a, rs_s = self._calc_stats(rs_results)
                rs_b_evals, rs_w_evals, rs_a_evals, rs_s_evals = self._calc_stats(rs_evals)
                rs_best_sequence = self._find_best_sequence(rs_sequences, rs_results)

                print("End of computation for RandomSearch")

                # Greedy [1x]
                problem.reset_counter() # set to 0 the NFE counter
                t0_gr = time.perf_counter()
                greedy = Greedy(problem)
                best_greedy = greedy.run()
                t_gr = time.perf_counter() - t0_gr
                gr_b = float(best_greedy.fitness)
                gr_evals = problem.evaluations_count
                gr_best_sequence = best_greedy.sequence
                
                print("End of computation for Greedy")

                # Evolutionary Algorithm [10x]
                ea_results = []
                ea_evals = []
                ea_sequences = []
                t0_ea = time.perf_counter()
                for _ in range(runs_per_algo):
                    problem.reset_counter() # set to 0 the NFE counter
                    ea = EvolutionaryAlgorithm(problem, pop_size, generations-1, tour_size, px, pm, mutation, crossover, elitism)
                    best_ind = ea.run()
                    ea_results.append(float(best_ind.fitness))
                    ea_evals.append(problem.evaluations_count)
                    ea_sequences.append(best_ind.sequence)
                t_ea = time.perf_counter() - t0_ea
                ea_b, ea_w, ea_a, ea_s = self._calc_stats(ea_results)
                ea_b_evals, ea_w_evals, ea_a_evals, ea_s_evals = self._calc_stats(ea_evals)
                ea_best_sequence = self._find_best_sequence(ea_sequences, ea_results)

                print("End of computation for EA")

                # Simulated Annealing [10x]
                sa_results = []
                sa_evals = []
                sa_sequences = []
                t0_sa = time.perf_counter()
                for _ in range(runs_per_algo):
                    problem.reset_counter() # set to 0 the NFE counter
                    sa = SimulatedAnnealing(problem, total_evals, initial_temp, cooling_rate)
                    best_ind = sa.run()
                    sa_evals.append(problem.evaluations_count)
                    sa_results.append(float(best_ind.fitness))
                    sa_sequences.append(best_ind.sequence)
                t_sa = time.perf_counter() - t0_sa
                sa_b, sa_w, sa_a, sa_s = self._calc_stats(sa_results)
                sa_b_evals, sa_w_evals, sa_a_evals, sa_s_evals = self._calc_stats(sa_evals)
                sa_best_sequence = self._find_best_sequence(sa_sequences, sa_results)

                print("End of computation for SimulatedAnnealing")


                # display head of table
                
                
                print("-" * 135)
                print(f"{'Instance':<15} | {'Random Alg. [10k]':<26} | {'Greedy Alg.':<11} | {'Evol. Alg. [10x]':<26} | {'SA [10x]':<26} | {'Best known value'}")
                print(f"{nom_inst:<15} | {'best*   worst  avg    std':<26} | {'best*':<11} | {'best*   worst  avg    std':<26} | {'best*   worst  avg    std':<26} |")
                print("-" * 135)

                
                rs_str = f"{rs_b:>5.0f} {rs_w:>6.0f} {rs_a:>6.1f} {rs_s:>5.1f}"
                gr_str = f"{gr_b:>5.0f}"
                ea_str = f"{ea_b:>5.0f} {ea_w:>6.0f} {ea_a:>6.1f} {ea_s:>5.1f}"
                sa_str = f"{sa_b:>5.0f} {sa_w:>6.0f} {sa_a:>6.1f} {sa_s:>5.1f}"

                rs_evals_str = f"{rs_b_evals:>5.0f} {rs_w_evals:>6.0f} {rs_a_evals:>6.1f} {rs_s_evals:>5.1f}"
                gr_evals_str = f"{gr_evals:>5.0f}"
                ea_evals_str = f"{ea_b_evals:>5.0f} {ea_w_evals:>6.0f} {ea_a_evals:>6.1f} {ea_s_evals:>5.1f}"
                sa_evals_str = f"{sa_b_evals:>5.0f} {sa_w_evals:>6.0f} {sa_a_evals:>6.1f} {sa_s_evals:>5.1f}"

                print(f"{'Score':<15} | {rs_str:<26} | {gr_str:<11} | {ea_str:<26} | {sa_str:<26} | {best_known_value}")
                print(f"{'Time':<15} | {t_rs:>7.2f}s {'':<17} | {t_gr:>6.4f}s {'':<3} | {t_ea:>7.2f}s {'':<17} | {t_sa:>5.1f}s |")
                print(f"{'NFE':<15} | {rs_evals_str:<26} | {gr_evals_str:<11} | {ea_evals_str:<26} | {sa_evals_str:<26} |")
                print("-" * 135)

                row = [
                    instance, pop_size, generations, px, pm, tour_size, mutation, crossover, elitism, initial_temp, cooling_rate,
                    round(rs_b, 1), round(rs_w, 1), round(rs_a, 2), round(rs_s, 2), round(t_rs, 3), round(rs_a_evals, 1), rs_best_sequence,
                    round(gr_b, 1), round(t_gr, 3), gr_evals, gr_best_sequence,
                    round(ea_b, 1), round(ea_w, 1), round(ea_a, 2), round(ea_s, 2), round(t_ea, 3), round(ea_a_evals, 1), ea_best_sequence,
                    round(sa_b, 1), round(sa_w, 1), round(sa_a, 2), round(sa_s, 2), round(t_sa, 3), round(sa_a_evals, 1), sa_best_sequence,
                    best_known_value
                ]
                writer.writerow(row)
                
                # force saving
                file.flush()
                print(f"Instance {instance} saved!")