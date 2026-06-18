# app/Problem/Problem.py

import numpy as np

from app.Problem.DataLoader import DataLoader
from app.Problem.Evaluator import Evaluator, KNNEvaluator, RFEvaluator, SVMEvaluator
from app.Problem.Fitness import ErrorFitness, Fitness, WeightedErrorFitness
from app.utilities.config_loader import load_config


class Problem:
    """
    Represents the Feature Selection problem and contain the fitness function
    """

    def __init__(self, X_train, y_train, X_test, y_test, metadata, evaluator, fitness):
        # Training data
        self.X = X_train
        self.y = y_train
        self.y_values = y_train.values.ravel()

        # Test data
        self.X_test = X_test
        self.y_test = y_test
        self.y_test_values = y_test.values.ravel()

        self.metadata = metadata
        self.num_features = X_train.shape[1]
        self.num_instances = metadata.num_instances
        self.evaluator: Evaluator = evaluator
        self.fitness: Fitness = fitness
        self.evaluation_model = evaluator.name
        self.fitness_name = fitness.name
        self.alpha = getattr(fitness, "alpha", None)
        self.cv_folds = getattr(evaluator, "cv_folds", None)
        self.evaluations_count = 0

    def __str__(self):
        return (
            f"Problem: dataset with {self.num_features} "
            f"features and {self.num_instances} instances."
            f" Evaluator: {self.evaluation_model}, "
            f"Fitness: {self.fitness_name}"
        )

    def reset_counter(self):
        self.evaluations_count = 0

    @staticmethod
    def _build_evaluator(evaluation_model, cv_folds):
        if evaluation_model == "svm":
            return SVMEvaluator(cv_folds=cv_folds, scoring="accuracy")
        if evaluation_model == "knn":
            return KNNEvaluator(cv_folds=cv_folds, scoring="accuracy")
        if evaluation_model == "rf":
            return RFEvaluator(cv_folds=cv_folds)
        raise ValueError(f"Unknown evaluation model: {evaluation_model}")

    @staticmethod
    def _build_fitness(fitness_name, alpha):
        if fitness_name == "error":
            return ErrorFitness()
        if fitness_name == "weighted_error":
            return WeightedErrorFitness(alpha)
        raise ValueError(f"Unknown fitness: {fitness_name}")

    @classmethod
    def from_dataset(
        cls,
        X_train,
        X_test,
        y_train,
        y_test,
        metadata,
        evaluation_config=None,
    ):
        config = load_config()
        evaluation_config = evaluation_config or {}
        evaluation_model = evaluation_config.get(
            "evaluation_model",
            config.get("evaluation_model", "svm"),
        )
        cv_folds = evaluation_config.get(
            "cv_folds",
            config.get("cv_folds", 5),
        )
        fitness_name = evaluation_config.get(
            "fitness",
            config.get("fitness", "weighted_error"),
        )
        alpha = evaluation_config.get("alpha", config.get("alpha", 0.5))

        evaluator = cls._build_evaluator(evaluation_model, cv_folds)
        fitness = cls._build_fitness(fitness_name, alpha)

        return cls(X_train, y_train, X_test, y_test, metadata, evaluator, fitness)

    @classmethod
    def load_dataset(cls, dataset_id, evaluation_config=None):
        config = load_config()
        evaluation_config = evaluation_config or {}
        seed = evaluation_config.get(
            "seed",
            config.get("seed", 42),
        )
        loaded_data = DataLoader.load_data(dataset_id, random_state=seed)
        return cls.from_dataset(*loaded_data, evaluation_config=evaluation_config)

    def evaluate(self, feature_mask):
        """
        Compute the fitness function for a subset of features.
        feature_mask: binar list (ex: [1, 0, 1, 0...]) of size num_features
        """
        return float(self.evaluate_batch([feature_mask])[0])

    def evaluate_batch(self, feature_masks):
        masks = np.asarray(feature_masks, dtype=int)
        if masks.ndim == 1:
            masks = masks.reshape(1, -1)
        if masks.shape[1] != self.num_features:
            raise ValueError(
                f"Expected masks with {self.num_features} features, got {masks.shape[1]}."
            )

        self.evaluations_count += int(masks.shape[0])
        results = np.ones(masks.shape[0], dtype=float)
        feature_counts = masks.sum(axis=1)

        for row_index, mask in enumerate(masks):
            selected_indices = np.flatnonzero(mask)
            if selected_indices.size == 0:
                continue

            X_subset = self.X.iloc[:, selected_indices]
            accuracy = self.evaluator.evaluate(X_subset, self.y_values)
            error = 1.0 - accuracy
            feature_ratio = float(feature_counts[row_index]) / self.num_features
            results[row_index] = self.fitness.compute(error, feature_ratio)

        return results

    def evaluate_final(self, feature_mask):
        """Fit model on train data, score on test data (used after optimization)."""
        selected_indices = [i for i, bit in enumerate(feature_mask) if bit == 1]
        if len(selected_indices) == 0:
            return 0.0

        X_train_sub = self.X.iloc[:, selected_indices]
        X_test_sub = self.X_test.iloc[:, selected_indices]
        return self.evaluator.evaluate_final(
            X_train_sub,
            self.y_values,
            X_test_sub,
            self.y_test_values,
        )
