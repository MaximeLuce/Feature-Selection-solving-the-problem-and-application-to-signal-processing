import csv
import json
import os
import statistics
import timeit

import numpy as np

from app.MLModels.HeavyModelSVM import HeavyModelSVM
from app.Problem.Problem import Problem
from app.Utilities.ConfigLoader import load_config
from app.decoders import build_decoder
from app.differential_evolution.algorithm import DifferentialEvolution
from app.differential_evolution.featureselection import FeatureSelectionProblem


class DEParameters:
    def __init__(self):
        config = load_config()
        self.dataset_ids = config.get("dataset_ids", [0])
        self.runs_per_algo = config.get("runs_per_algo", 10)
        self.cases = config.get("cases_DE_decoding", [])

        if not self.cases:
            print("No DE cases config has been loaded.")

    def _calc_stats(self, results):
        best = max(results)
        worst = min(results)
        avg = statistics.mean(results)
        std = statistics.stdev(results) if len(results) > 1 else 0.0
        return best, worst, avg, std

    def _build_de_config(self, case):
        return {
            "popsize": case["popsize"],
            "max_generations": case["max_generations"],
            "seed": case["seed"],
            "strategy": case.get("strategy", "rand/1"),
            "F1": case.get("F1", 0.5),
            "F2": case.get("F2", 0.5),
            "CR": case.get("CR", 0.7),
        }

        
    def run_all(self):
        csv_filepath = "app/Results/SAParameters/DEParameters.csv"

        print(f"Start running... Results will be saved to {csv_filepath}")

        file_exists = os.path.isfile(csv_filepath)
        with open(csv_filepath, mode="a", newline="") as file:
            writer = csv.writer(file, delimiter=";")

            if not file_exists:
                header = [
                    "Case_ID",
                    "Dataset_ID",
                    "Population_size",
                    "Generations_number",
                    "Seed",
                    "Decoder",
                    "Decoder_Params",
                    "DE_Best_light",
                    "DE_Worst_light",
                    "DE_Avg_light",
                    "DE_Std_light",
                    "DE_Best_heavy",
                    "DE_Worst_heavy",
                    "DE_Avg_heavy",
                    "DE_Std_heavy",
                    "DE_Time(s)",
                    "DE_NFE_Avg",
                    "Champion_SVM",
                    "Nb_Features_Keep",
                ]
                writer.writerow(header)
                
            for dataset_id in self.dataset_ids:
                try:
                    problem = Problem.load_dataset(dataset_id)
                    print(
                        f"Problem loaded: ID={dataset_id} with {problem.num_features} "
                        f"features and {problem.num_instances} instances."
                    )
                    heavy_model = HeavyModelSVM(dataset_id)
                except Exception as exc:
                    print(f"Error loading data: {exc}")
                    continue

                print(f"\n{'=' * 50}")
                print(f"=== ANALYZING DATASET ID {dataset_id} ===")
                print(f"{'=' * 50}")
                
                for case in self.cases:
                    decoder_config = case.get("decoder")
                    decoder = build_decoder(decoder_config)
                    de_config = self._build_de_config(case)

                    print(f"\n--- RUNNING CASE {case['id']} ---")
                    print(f"Decoder: {decoder_config['name']}")

                    de_results_light = []
                    de_results_heavy = []
                    de_evals = []

                    best_overall_fitness = float("inf")
                    best_overall_mask = np.ones(problem.num_features, dtype=int)

                    start_time = timeit.default_timer()
                    seed = de_config["seed"]
                    for i in range(self.runs_per_algo):
                        
                        de_config["seed"] = seed ^ (i * 10000001)
                        de_problem = FeatureSelectionProblem(problem, decoder)
                        de = DifferentialEvolution(de_problem, de_config)
                        result = de.run()

                        best_mask = de_problem.decode(result.best)
                        de_evals.append(de_problem.evaluations_count)

                        score_light = (1.0 - result.best_fitness) * 100
                        score_heavy = heavy_model.evaluate(best_mask) * 100

                        de_results_light.append(score_light)
                        de_results_heavy.append(score_heavy)

                        if result.best_fitness < best_overall_fitness:
                            best_overall_fitness = result.best_fitness
                            best_overall_mask = best_mask.copy()

                    elapsed = timeit.default_timer() - start_time
                    de_b, de_w, de_a, de_s = self._calc_stats(de_results_light)
                    de_b_heavy, de_w_heavy, de_a_heavy, de_s_heavy = self._calc_stats(
                        de_results_heavy
                    )
                    de_b_evals, de_w_evals, de_a_evals, de_s_evals = self._calc_stats(
                        de_evals
                    )

                    score_champion_heavy = heavy_model.evaluate(best_overall_mask) * 100
                    nb_features_keep = int(best_overall_mask.sum())

                    print("-" * 135)
                    print(
                        f"{'Dataset_ID':<15} | {'DE (Light Model) [Nx]':<27} | "
                        f"{'DE (Heavy Model) [Nx]':<26} | {'Champion SVM'}"
                    )
                    print(
                        f"{dataset_id:<15} |  {'best*   worst  avg    std':<26} | "
                        f"{'best*   worst  avg    std':<26} | "
                        f"{score_champion_heavy:.2f}% ({nb_features_keep} features)"
                    )
                    print("-" * 135)

                    de_str = f"{de_b:>5.0f} {de_w:>6.0f} {de_a:>6.1f} {de_s:>5.1f}"
                    de_str_heavy = (
                        f"{de_b_heavy:>5.0f} {de_w_heavy:>6.0f} "
                        f"{de_a_heavy:>6.1f} {de_s_heavy:>5.1f}"
                    )
                    de_evals_str = (
                        f"{de_b_evals:>5.0f} {de_w_evals:>6.0f} "
                        f"{de_a_evals:>6.1f} {de_s_evals:>5.1f}"
                    )

                    print(f"{'Score':<15} | {de_str:<26} | {de_str_heavy:<26}")
                    print(f"{'Time':<15} |  {elapsed:<5.1f}s |")
                    print(f"{'NFE':<15} | {de_evals_str:<26} |")
                    print("-" * 135)

                    row = [
                        case["id"],
                        dataset_id,
                        de_config["popsize"],
                        de_config["max_generations"],
                        de_config["seed"],
                        decoder_config["name"],
                        json.dumps(decoder_config, sort_keys=True),
                        round(de_b, 1),
                        round(de_w, 1),
                        round(de_a, 2),
                        round(de_s, 2),
                        round(de_b_heavy, 1),
                        round(de_w_heavy, 1),
                        round(de_a_heavy, 2),
                        round(de_s_heavy, 2),
                        round(elapsed, 3),
                        round(de_a_evals, 1),
                        round(score_champion_heavy, 2),
                        nb_features_keep,
                    ]
                    writer.writerow(row)
                    file.flush()
                    print(f"Case {case['id']} saved!")


if __name__ == "__main__":
    DEParameters().run_all()
