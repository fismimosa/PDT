import os
import os
# Set environment variables for controlling the number of threads in certain libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import pandas as pd
import numpy as np
import time
from RuleTree import RuleTreeClassifier
from RuleTree.stumps.instance_stumps import * 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score


import pandas as pd
import numpy as np
import time
import math
from itertools import product
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from RuleTree import RuleTreeClassifier
from RuleTree.stumps.instance_stumps import * 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
pt_stump = pt_stump_call()


pt_parameters = {               'max_depth': [5],
                                'random_state' : [42],
                                'base_stumps' : [[pt_stump]],
                                'stump_selection' : ['best'],
                                'prune_useless_leaves' : [False],   
                            }


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

# Function to compute classification metrics
def compute_metrics_classifier(y_true, y_pred, prefix):
    return {
        f'{prefix}_accuracy': accuracy_score(y_true, y_pred),
        f'{prefix}_balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        f'{prefix}_precision_macro': precision_score(y_true, y_pred, average='macro'),
        f'{prefix}_precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        f'{prefix}_recall_macro': recall_score(y_true, y_pred, average='macro'),
        f'{prefix}_recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        f'{prefix}_f1_macro': f1_score(y_true, y_pred, average='macro'),
        f'{prefix}_f1_weighted': f1_score(y_true, y_pred, average='weighted'),
    }

# Dictionary to store datasets
datasets = {}

# Directory containing datasets
datasets_folder = "sensitivity_datasets"

# Load datasets dynamically
if os.path.exists(datasets_folder):
    for file in os.listdir(datasets_folder):
        if file.endswith(".csv"):  # Ensure only CSV files are processed
            dataset_name = os.path.splitext(file)[0]  # Remove .csv extension
            datasets[dataset_name] = pd.read_csv(os.path.join(datasets_folder, file))

print(f"Loaded {len(datasets)} datasets: {list(datasets.keys())}")

# List to store results
total_res = []

# Iterate over datasets
for dataset_name, dataframe in datasets.items():
    print(f"Processing dataset: {dataset_name}")

    

    X = dataframe.drop(columns=['label']).values
    y = np.array(dataframe.label)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Fit and transform on training data
    X_test = scaler.transform(X_test)  # Only transform test data

    model_result = {'dataset_name': dataset_name}

    pt_model = RuleTreeClassifier(max_depth=5, distance_measure='euclidean', 
                                  random_state = 42, base_stumps = [pt_stump], 
                                  stump_selection = 'best')

    
    _, train_time = measure_time(pt_model.fit, X_train, y_train)
    model_result.update({f'PT_train_time_{k}': v for k, v in train_time.items()})

    print(pt_model.root.stump.threshold_original)

    
    y_pred_train, predict_train_time = measure_time(pt_model.predict, X_train)
    model_result.update({f'PT_predict_on_train_time_{k}': v for k, v in predict_train_time.items()})

    y_pred_test, predict_test_time = measure_time(pt_model.predict, X_test)
    model_result.update({f'PT_predict_on_test_time_{k}': v for k, v in predict_test_time.items()})

    model_result.update(compute_metrics_classifier(y_train, y_pred_train, f'PT_measure_train'))
    model_result.update(compute_metrics_classifier(y_test, y_pred_test, f'PT_measure_test'))

    total_res.append(model_result)

# Convert results to DataFrame
df = pd.DataFrame(total_res)

# Compute mean and standard deviation for each column
mean_row = df.mean(numeric_only=True).to_frame().T
std_row = df.std(numeric_only=True).to_frame().T

# Add labels for identification
mean_row.insert(0, df.columns[0], "mean")
std_row.insert(0, df.columns[0], "std")

# Append mean and std rows to DataFrame
df = pd.concat([df, mean_row, std_row], ignore_index=True)

# Save to CSV
df.to_csv('pivottree_depth5_results_table.csv', index=False)

print("File 'pivottree_depth5_results_table.csv' saved with mean and std rows!")
