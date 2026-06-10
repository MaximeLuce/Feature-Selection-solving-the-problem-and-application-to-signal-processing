# problem.py

from app.Problem.DataLoader import DataLoader
from app.Problem.Evaluator import Evaluator, KNNEvaluator, SVMEvaluator
from app.Problem.Fitness import ErrorFitness, Fitness, WeightedErrorFitness
from app.Utilities.ConfigLoader import load_config


class Problem:
    """
    Represents the Feature Selection problem and contain the fitness function
    """

    def __init__(self, X, y, metadata, evaluator, fitness):
        self.X = X
        self.y = y
        self.metadata = metadata
        self.num_features = X.shape[1]
        self.num_instances = metadata.num_instances
        self.evaluator: Evaluator = evaluator
        self.fitness: Fitness = fitness
        self.evaluation_model = evaluator.name
        self.fitness_name = fitness.name
        self.alpha = getattr(fitness, "alpha", None)
        self.cv_folds = getattr(evaluator, "cv_folds", None)
        self.y_values = self.y.values.ravel()
        self.evaluations_count = 0

    def __str__(self):
        return (
            f"Problem loaded: dataset with {self.num_features} "
            f"features and {self.num_instances} instances."
            f" Evaluator: {self.evaluation_model}, "
            f"Fitness: {self.fitness_name}"
        )

    def reset_counter(self):
        self.evaluations_count = 0

    @classmethod
    def load_dataset(cls, dataset_id, evaluation_config=None):
        config = load_config()
        evaluation_config = evaluation_config or {}
        evaluation_model = evaluation_config.get(
            "evaluation_model",
            config.get("evaluation_model", "svm"),
        )
        fitness_name = evaluation_config.get(
            "fitness",
            config.get("fitness", "weighted_error"),
        )

        X, y, metadata = DataLoader.load_data(dataset_id)
        if evaluation_model == "svm":
            evaluator = SVMEvaluator()
        elif evaluation_model == "knn":
            evaluator = KNNEvaluator()
        else:
            raise ValueError(f"Unknown evaluation model: {evaluation_model}")

        if fitness_name == "error":
            fitness = ErrorFitness()
        elif fitness_name == "weighted_error":
            alpha = evaluation_config.get("alpha", config.get("alpha", 0.5))
            fitness = WeightedErrorFitness(alpha)
        else:
            raise ValueError(f"Unknown fitness: {fitness_name}")

        return cls(X, y, metadata, evaluator, fitness)

    def evaluate(self, feature_mask):
        """
        Compute the fitness function for a subset of features.
        feature_mask: binar list (ex: [1, 0, 1, 0...]) of size num_features
        """
        self.evaluations_count += 1

        selected_indices = [i for i, bit in enumerate(feature_mask) if bit == 1]

        if len(selected_indices) == 0:
            return 1.0

        X_subset = self.X.iloc[:, selected_indices]
        accuracy = self.evaluator.evaluate(X_subset, self.y_values)
        error = 1.0 - accuracy
        feature_ratio = len(selected_indices) / self.num_features
        return self.fitness.compute(error, feature_ratio)
