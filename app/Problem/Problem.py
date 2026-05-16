import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from app.Problem.DataLoader import * # use in the alternative constructor

class Problem:
    """
    Represents the Feature Selection problem and contain the fitness function
    """
    
    def __init__(self, X, y, metadata, k=6):
        self.X = X
        self.y = y
        self.num_features = X.shape[1] # total number of columns=features
        self.num_instances = metadata.num_instances

        # light model for heuristique
        self.model = KNeighborsClassifier(n_neighbors=k) 
        self.evaluations_count = 0

    def __str__(self):
        return f'Problem loaded: dataset with {self.num_features} features and {self.num_instances} instances.'
    
    def reset_counter(self):
        """Reset evaluations_count counter to zéro."""
        self.evaluations_count = 0
        
    @classmethod
    def load_dataset(self, dataset_id, k=3):
        """Alternative constructor to instanciate from dataset."""
        X, y, metadata = DataLoader.load_data(dataset_id)
        return self(X, y, metadata, k)

    def evaluate(self, feature_mask):
        """
        Compute the fitness function for a subset of features.
        feature_mask: binar list (ex: [1, 0, 1, 0...]) of size num_features
        """
        self.evaluations_count += 1
        
        # get the index of selected features
        selected_indices = [i for i, bit in enumerate(feature_mask) if bit == 1]
        
        # if no selected feature, we render maximal penality
        if len(selected_indices) == 0:
            return 1.0 
            
        # filter the dataset to keep selected features
        X_subset = self.X.iloc[:, selected_indices]
        
        # formatting y for scikit-learn (debugging))
        y_values = self.y.values.ravel()
        
        # crossed validation evaluation (5-fold)
        scores = cross_val_score(self.model, X_subset, y_values, cv=5, scoring='accuracy')
        mean_accuracy = scores.mean()
        
        # we return the error (1 - accuracy) because using PFSP tools, we optimize by trying to get minimal value
        error = 1.0 - mean_accuracy
        
        return error


