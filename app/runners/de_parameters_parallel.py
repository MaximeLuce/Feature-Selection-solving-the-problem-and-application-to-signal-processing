# de_parameters_parallel.py

import csv
import json
import os
import random
import statistics
import timeit
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from app.decoders import build_decoder
from app.differential_evolution.algorithm import DifferentialEvolution
from app.differential_evolution.featureselection import FeatureSelectionProblem
from app.MLModels.HeavyModelSVM import HeavyModelSVM
from app.Problem.Problem import Problem
from app.Utilities.ConfigLoader import load_config

CSV_COLUMNS = [
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


class DEParametersParallell:
    def __init__(self):
        config = load_config()
        self.dataset_ids = config.get("dataset_ids", [0])
        self.runs_per_algo = config.get("runs_per_algo", 10)
        self.cases = config.get("cases_DE_strategy", [])
        self.csv_filepath = config.get(
            "csv_filepath",
            "app/Results/SAParameters/DEParameters_strategy.csv",
        )

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

    def _build_runs(self):
        run_id = 0
        for dataset_id in self.dataset_ids:
            for case in self.cases:
                base_seed = case["seed"]
                for i in range(self.runs_per_algo):
                    config = self._build_de_config(case)
                    config["seed"] = base_seed ^ (i * 10000001)
                    config["dataset_id"] = dataset_id
                    config["case_id"] = case["id"]
                    config["decoder"] = case["decoder"]
                    config["run_id"] = run_id
                    run_id += 1
                    yield config

    @staticmethod
    def _run_config(config: dict):
        print(f"\n--- RUNNING CASE {config['case_id']} ---")

        dataset_id = config["dataset_id"]
        problem = Problem.load_dataset(dataset_id)
        print(
            f"Problem loaded: ID={dataset_id} with {problem.num_features} "
            f"features and {problem.num_instances} instances."
        )

        decoder = build_decoder(config["decoder"])
        start_time = timeit.default_timer()
        de_problem = FeatureSelectionProblem(problem, decoder)
        de = DifferentialEvolution(de_problem, config)
        result = de.run()
        elapsed = timeit.default_timer() - start_time

        return {
            "run_id": config["run_id"],
            "case_id": config["case_id"],
            "dataset_id": dataset_id,
            "popsize": config["popsize"],
            "max_generations": config["max_generations"],
            "seed": config["seed"],
            "decoder_name": decoder.name,
            "decoder_params": config["decoder"],
            "best_vector": np.asarray(result.best, dtype=float),
            "best_fitness": float(result.best_fitness),
            "elapsed": elapsed,
            "evaluations_count": de_problem.evaluations_count,
        }

    def _build_csv_row(self, values: dict):
        row = {
            "Case_ID": values["case_id"],
            "Dataset_ID": values["dataset_id"],
            "Population_size": values["popsize"],
            "Generations_number": values["max_generations"],
            "Seed": values["seed"],
            "Decoder": values["decoder_name"],
            "Decoder_Params": json.dumps(values["decoder_params"], sort_keys=True),
            "DE_Best_light": round(values["de_b"], 1),
            "DE_Worst_light": round(values["de_w"], 1),
            "DE_Avg_light": round(values["de_a"], 2),
            "DE_Std_light": round(values["de_s"], 2),
            "DE_Best_heavy": round(values["de_b_heavy"], 1),
            "DE_Worst_heavy": round(values["de_w_heavy"], 1),
            "DE_Avg_heavy": round(values["de_a_heavy"], 2),
            "DE_Std_heavy": round(values["de_s_heavy"], 2),
            "DE_Time(s)": round(values["elapsed"], 3),
            "DE_NFE_Avg": round(values["de_a_evals"], 1),
            "Champion_SVM": round(values["score_champion_heavy"], 2),
            "Nb_Features_Keep": values["nb_features_keep"],
        }
        return self._check_csv_row(row)

    def _check_csv_row(self, row: dict):
        missing = [column for column in CSV_COLUMNS if column not in row]
        if missing:
            raise ValueError(f"Missing CSV columns: {missing}")

        return {column: row[column] for column in CSV_COLUMNS}

    def _print_summary(self, row: dict):
        print("-" * 135)
        print(
            f"{'Dataset_ID':<15} | {'DE (Light Model) [Nx]':<27} | "
            f"{'DE (Heavy Model) [Nx]':<26} | {'Champion SVM'}"
        )
        print(
            f"{row['Dataset_ID']:<15} |  {'best*   worst  avg    std':<26} | "
            f"{'best*   worst  avg    std':<26} | "
            f"{row['Champion_SVM']:.2f}% ({row['Nb_Features_Keep']} features)"
        )
        print("-" * 135)

        de_str = (
            f"{row['DE_Best_light']:>5.0f} {row['DE_Worst_light']:>6.0f} "
            f"{row['DE_Avg_light']:>6.1f} {row['DE_Std_light']:>5.1f}"
        )
        de_str_heavy = (
            f"{row['DE_Best_heavy']:>5.0f} {row['DE_Worst_heavy']:>6.0f} "
            f"{row['DE_Avg_heavy']:>6.1f} {row['DE_Std_heavy']:>5.1f}"
        )
        de_evals_str = f"{row['DE_NFE_Avg']:>6.1f}"

        print(f"{'Score':<15} | {de_str:<26} | {de_str_heavy:<26}")
        print(f"{'Time':<15} |  {row['DE_Time(s)']:<5.1f}s |")
        print(f"{'NFE Avg':<15} | {de_evals_str:<26} |")
        print("-" * 135)

    def _aggregate_results(self, results: list[dict]):
        if not results:
            return []

        data = pd.DataFrame(results).sort_values("run_id")
        rows = []

        for _, group in data.groupby(["dataset_id", "case_id"], sort=False):
            first = group.iloc[0]
            last = group.iloc[-1]
            dataset_id = int(first["dataset_id"])
            seed = int(last["seed"])
            decoder_config = first["decoder_params"]
            decoder_name = first["decoder_name"]

            problem = Problem.load_dataset(dataset_id)
            heavy_model = HeavyModelSVM(dataset_id)
            decoder = build_decoder(decoder_config)
            de_problem = FeatureSelectionProblem(problem, decoder)

            de_results_light = []
            de_results_heavy = []
            de_evals = []
            best_overall_fitness = float("inf")
            best_overall_mask = np.ones(problem.num_features, dtype=int)
            elapsed_total = 0.0

            for run in group.to_dict("records"):
                best_vector = np.asarray(run["best_vector"], dtype=float)
                best_mask = de_problem.decode(best_vector)

                score_light = (1.0 - float(run["best_fitness"])) * 100
                score_heavy = heavy_model.evaluate(best_mask) * 100

                de_results_light.append(score_light)
                de_results_heavy.append(score_heavy)
                de_evals.append(run["evaluations_count"])
                elapsed_total += float(run["elapsed"])

                if float(run["best_fitness"]) < best_overall_fitness:
                    best_overall_fitness = float(run["best_fitness"])
                    best_overall_mask = best_mask.copy()

            de_b, de_w, de_a, de_s = self._calc_stats(de_results_light)
            de_b_heavy, de_w_heavy, de_a_heavy, de_s_heavy = self._calc_stats(
                de_results_heavy
            )
            _, _, de_a_evals, _ = self._calc_stats(de_evals)

            score_champion_heavy = heavy_model.evaluate(best_overall_mask) * 100
            nb_features_keep = int(best_overall_mask.sum())

            row = self._build_csv_row(
                {
                    "case_id": first["case_id"],
                    "dataset_id": dataset_id,
                    "popsize": int(first["popsize"]),
                    "max_generations": int(first["max_generations"]),
                    "seed": seed,
                    "decoder_name": decoder_name,
                    "decoder_params": decoder_config,
                    "de_b": de_b,
                    "de_w": de_w,
                    "de_a": de_a,
                    "de_s": de_s,
                    "de_b_heavy": de_b_heavy,
                    "de_w_heavy": de_w_heavy,
                    "de_a_heavy": de_a_heavy,
                    "de_s_heavy": de_s_heavy,
                    "elapsed": elapsed_total,
                    "de_a_evals": de_a_evals,
                    "score_champion_heavy": score_champion_heavy,
                    "nb_features_keep": nb_features_keep,
                }
            )

            rows.append(row)

        return rows

    def _write_csv(self, rows: list[dict]):
        file_exists = os.path.isfile(self.csv_filepath)
        with open(self.csv_filepath, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS, delimiter=";")

            if not file_exists:
                writer.writeheader()

            for row in rows:
                writer.writerow(self._check_csv_row(row))
                file.flush()
                print(f"Case {row['Case_ID']} saved!")

    def run_all(self):
        print(f"Start running... Results will be saved to {self.csv_filepath}")

        configurations = list(self._build_runs())
        if not configurations:
            return

        random.shuffle(configurations)
        num_workers = os.cpu_count() or 4

        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self._run_config, config): config["run_id"]
                for config in configurations
            }

            for future in as_completed(futures):
                run_id = futures[future]
                results.append(future.result())
                print(f"Run {run_id} done: {len(results)}/{len(configurations)}")

        rows = self._aggregate_results(results)
        for row in rows:
            self._print_summary(row)
        self._write_csv(rows)


if __name__ == "__main__":
    DEParametersParallell().run_all()
