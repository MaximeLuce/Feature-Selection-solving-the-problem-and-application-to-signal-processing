from types import SimpleNamespace

import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


class DataLoader:
    _cache = {}

    @staticmethod
    def load_data(dataset_id, test_size=0.2, random_state=42, verbose=False):
        cache_key = (dataset_id, test_size, random_state)
        if cache_key in DataLoader._cache:
            if verbose:
                print(f"Dataset cache hit: ID={dataset_id}")
            return DataLoader._cache[cache_key]

        if verbose:
            print(f"Dataset loading: ID={dataset_id}")
        if dataset_id == 0:
            file_path = "app/Data/sonar_dataset.csv"
            df = pd.read_csv(file_path)
            df = df[df.iloc[:, -1] != "Debris"]
            X = df.filter(like="freq_")
            y = df.iloc[:, -1]
            metadata = SimpleNamespace(
                name="Sonar Dataset - Kaggle (Local)",
                source="local",
                file_path=file_path,
                shape=df.shape,
                features_count=X.shape[1],
                num_instances=df.shape[0],
            )
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
        if verbose:
            print(f"Dataset loaded: ID={dataset_id}")
        return loaded_data
