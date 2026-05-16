import time
import sys
import os

from app.MLModels.HeavyModelSVM import HeavyModelSVM

def run_test():
    print("Heavy model (SVM) test starting...")

    print("Loading Data and SVM model creation and dataset id=151...")
    start_load = time.time()
    
    try:
        heavy_model = HeavyModelSVM(151)
        print(heavy_model)
    except Exception as e:
        print(f"Error loading data and model : {e}")
        return

    print("Starting evaluating the model on all features...")
    start_eval = time.time()
    
    baseline_accuracy = heavy_model.get_baseline() 
    
    end_eval = time.time()

    print("Results...")
    
    print(f"Execution time : {end_eval - start_eval:.2f} secondes")
    print(f"Accuracy: {baseline_accuracy * 100:.2f}%")
    print(f"Used features: {heavy_model.num_features} (100% of the dataset)")
    
if __name__ == "__main__":
    run_test()