from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split

# to format dic as UCIrepo
from types import SimpleNamespace


class DataLoader:
    _cache = {}

    """
    Utilitary to read and extract the data from the dataset
    Input: a specific ID for each UCI dataset (and 0 for local kaggle dataset)
    Output: - X_train, X_test ; train/test split of features
            - y_train, y_test ; train/test split of target
            - metadata ; information on the dataset
    """

    @staticmethod
    def load_data(dataset_id, test_size=0.2, random_state=42):
        """
        Load dataset and split into train/test.
        If id=0, we use the local kaggle data set.
        """
        cache_key = (dataset_id, test_size, random_state)
        if cache_key in DataLoader._cache:
            return DataLoader._cache[cache_key]

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
                "num_instances": df.shape[0],
            }
            metadata = SimpleNamespace(**metadata_dict) # formatting the dict into UCI object like

        else:
            dataset = fetch_ucirepo(id=dataset_id) 
            
            X = dataset.data.features 
            y = dataset.data.targets 
            metadata = dataset.metadata

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        metadata.num_instances = len(X_train)

        loaded_data = (X_train, X_test, y_train, y_test, metadata)
        DataLoader._cache[cache_key] = loaded_data
        return loaded_data
    
# TEST
'''
e = Emprunt(3,5)
print(e)

print(e.get_numero_lecteur())
print(e.get_numero_livre())
print(e.get_date())
'''