import numpy as np
from ucimlrepo import fetch_ucirepo

# Kaggle
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

# to format dic as UCIrepo
from types import SimpleNamespace

class DataLoader:
    """
    Utilitary to read and extract the data from the dataset
    Input: a specific ID for each UCI dataset (and 0 for local kaggle dataset)
    Output: - X ; all the features
            - y ; the cible
            - metadata ; information on the dataset
    """
    
    @staticmethod
    def load_data(dataset_id):
        """
        Get the dataset with id = dataset_id from UCI and split features (X) from cible (y).
        If id=0, we use the local kaggle data set.
        """
        print(f"Dataset ID={dataset_id} loading...")
        if dataset_id == 0:
            print("Loading local kaggle dataset...")
            file_path = "app/Data/sonar_dataset.csv"
            
            try:
                df = pd.read_csv(file_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"File {file_path} not found.")

            # removing Debris to transform the dataset into binar dataset
            df = df[df.iloc[:, -1] != 'Debris']
			# extract the features X and the cible y
            X = df.iloc[:, :-1] # all lines, all columns except the last one
            y = df.iloc[:, -1] # all lines, just the last column
            
            # metadata using the UCI format
            metadata_dict = {
                "name": "Sonar Dataset - Kaggle (Local)",
                "source": "local",
                "file_path": file_path,
                "shape": df.shape,
                "features_count": X.shape[1],
				"num_instances": df.shape[0]
            }
            metadata = SimpleNamespace(**metadata_dict) # formatting the dict into UCI object like
            # countingprint(df['object_class'].value_counts())

            return X, y, metadata
            
        else:
            dataset = fetch_ucirepo(id=dataset_id) 
            
            X = dataset.data.features 
            y = dataset.data.targets 
            metadata = dataset.metadata
            
            return X, y, metadata
    
# TEST
'''
e = Emprunt(3,5)
print(e)

print(e.get_numero_lecteur())
print(e.get_numero_livre())
print(e.get_date())
'''