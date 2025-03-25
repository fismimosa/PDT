# import concurrent
# import itertools
# import json
# import multiprocessing
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    f1_score,
    recall_score,
    accuracy_score,
    precision_score,
    balanced_accuracy_score,
)

from pdll import PairwiseDifferenceClassifier

# Function to measure execution time
def measure_time(func, *args, **kwargs):
    """Measures wall, process, and performance time of a function."""
    start_time_wall, start_time_process, start_time_perf = time.time(), time.process_time(), time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_times = {
        'wall': time.time() - start_time_wall,
        'process': time.process_time() - start_time_process,
        'perf': time.perf_counter() - start_time_perf
    }
    return result, elapsed_times
 
def compute_metrics_classifier(y_true, y_pred, prefix):
    """
    Computes classification metrics including accuracy, balanced accuracy, precision, recall, and F1-score.
    Includes weighted variations for multi-class classification.
 
    Parameters:
    - y_true (array-like): True labels
    - y_pred (array-like): Predicted labels
    - prefix (str): A prefix for metric names to differentiate multiple models or settings.
 
    Returns:
    - dict: A dictionary containing the computed metrics.
    """
    
    return {
        f'accuracy': accuracy_score(y_true, y_pred),
        f'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        f'precision_macro': precision_score(y_true, y_pred, average='macro'),
        f'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        f'recall_macro': recall_score(y_true, y_pred, average='macro'),
        f'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        f'f1_macro': f1_score(y_true, y_pred, average='macro'),
        f'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
    }

dataset_path = 'sensitivity_dataset'
results = []

for filename in os.listdir(dataset_path):
    if filename.endswith('.csv'):
        dataset_name = os.path.splitext(filename)[0]
        file_path = os.path.join(dataset_path, filename)
        print(f"Process dataset: {dataset_name}")
        
        dataframe = pd.read_csv(file_path)
        X = dataframe.drop(columns = ['label']).values
        y = np.array(dataframe.label)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # clf = KNeighborsClassifier(algorithm='brute') # delete comment here to use standard KNN otherwise use pdl as below
        clf = PairwiseDifferenceClassifier(estimator=KNeighborsClassifier(algorithm='brute'))
        
        # Fit the model using the train matrix
        clf.fit(X_train, y_train)
        _, train_time = measure_time(clf.fit, X_train, y_train)
            
        # Predictions on training and test sets
        y_pred_train, predict_train_time = measure_time(clf.predict, X_train)
        y_pred_test, predict_test_time = measure_time(clf.predict, X_test)
        
        # Store prediction results
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        # Update dataset entry dictionary
        row = {"dataset": dataset_name}
        row.update({f'train_time_{k}': v for k, v in train_time.items()})
        row.update({f'predict_on_train_time_{k}': v for k, v in predict_train_time.items()})
        row.update({f'predict_on_test_time_{k}': v for k, v in predict_test_time.items()})
        row.update(compute_metrics_classifier(y_test, y_pred_test, dataset_name)) 
        
        # Append dataset entry dictionary to the results list
        results.append(row)

# save results as pandas df
df_results = pd.DataFrame(results)

# Add avg and std rows
numeric_cols = df_results.select_dtypes(include=['number']).columns

avg_row = df_results[numeric_cols].mean()
std_row = df_results[numeric_cols].std()

avg_row["dataset"] = "avg"
std_row["dataset"] = "std"

avg_df = avg_row.to_frame().T
std_df = std_row.to_frame().T

df_results = pd.concat([df_results, avg_df, std_df], ignore_index=True)

# export to csv
df_results.to_csv('experiments_new/kNN5_pdl_brute_sensitivity_latest.csv', index=False)