import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from app.Problem.DataLoader import DataLoader

class HeavyModelSVM:
    def __init__(self, dataset_id):
        self.X, self.y, self.metadata = DataLoader.load_data(dataset_id)
        self.num_features = self.X.shape[1]
        
        # SVM with StandardScaler
        self.model = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='scale'))
        
        # crossed validation stratified and fixed for reproductivility
        self.cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def __str__(self):
        return (f"Heavy model loaded on model with {self.metadata.num_features} features on {self.metadata.num_instances} instances.")

    def evaluate(self, feature_mask=None):
        if feature_mask is None:
            feature_mask = [1] * self.num_features
            
        selected_indices = [i for i, bit in enumerate(feature_mask) if bit == 1]
        
        if len(selected_indices) == 0:
            return 0.0
            
        X_subset = self.X.iloc[:, selected_indices]
        y_values = self.y.values.ravel()
     
        scores = cross_val_score(self.model, X_subset, y_values, cv=self.cv_strategy, scoring='accuracy')
        
        return scores.mean()

    def get_baseline(self):
        return self.evaluate(feature_mask=None)