from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class Evaluator:
    name = "evaluator"
    
    def evaluate(self, X_subset, y_values) -> float:
        """Cross-validated accuracy on training data."""
        raise NotImplementedError

    def evaluate_final(self, X_train, y_train, X_test, y_test) -> float:
        """Fit on train, score on test."""
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

    def evaluate_final(self, X_train, y_train, X_test, y_test):
        model = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale"))
        model.fit(X_train, y_train)
        return float(model.score(X_test, y_test))


class KNNEvaluator(Evaluator):
    name = "knn"

    def __init__(self, cv_folds=5, scoring="accuracy", n_neighbors=3):
        self.cv_folds = cv_folds
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors)
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

    def evaluate_final(self, X_train, y_train, X_test, y_test):
        model = KNeighborsClassifier(self.n_neighbors)
        model.fit(X_train, y_train)
        return float(model.score(X_test, y_test))


class RFEvaluator(Evaluator):
    name = "rf"

    def __init__(self, cv_folds=5, scoring="accuracy", n_estimators=100):
        self.cv_folds = cv_folds
        self.n_estimators = n_estimators
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        self.cv_strategy = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=42,
        )
        self.scoring = scoring

    def evaluate(self, X_subset, y_values):
        scores = cross_val_score(
            self.model, X_subset, y_values,
            cv=self.cv_strategy, scoring=self.scoring,
        )
        return float(scores.mean())

    def evaluate_final(self, X_train, y_train, X_test, y_test):
        model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=42)
        model.fit(X_train, y_train)
        return float(model.score(X_test, y_test))
