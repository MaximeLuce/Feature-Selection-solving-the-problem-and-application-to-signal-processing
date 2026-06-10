from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class Evaluator:
    name = "evaluator"
    
    def evaluate(self, X_subset, y_values) -> float:
        raise NotImplementedError

class SVMEvaluator(Evaluator):
    name = "svm"

    def __init__(self, cv_folds=5, scoring="accuracy"):
        self.cv_folds = cv_folds
        self.model = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", gamma="scale"),
        )
        self.cv_strategy = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=42,
        )
        self.scoring = scoring

    def evaluate(self, X_subset, y_values):
        scores = cross_val_score(
            self.model,
            X_subset,
            y_values,
            cv=self.cv_strategy,
            scoring=self.scoring
        )
        return float(scores.mean())


class KNNEvaluator(Evaluator):
    name = "knn"

    def __init__(self, cv_folds=5, scoring="accuracy", n_neighbors=3):
        self.cv_folds = cv_folds
        self.model = KNeighborsClassifier(n_neighbors)
        self.cv_strategy = cv_folds
        self.scoring = scoring

    def evaluate(self, X_subset, y_values):
        scores = cross_val_score(
            self.model,
            X_subset,
            y_values,
            cv=self.cv_strategy,
            scoring=self.scoring
        )
        return float(scores.mean())

