
import numpy

import numpy as np
import pandas as pd
import time
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import euclidean
from scipy.stats import ks_2samp
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import numpy as np
import time
import math
from itertools import product
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score




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


def compute_metrics(y_true, y_pred, prefix):
                        return {f'{prefix}_{metric}': value for metric, value in zip(
                            ['mse', 'rmse', 'mae', 'r2'],
                            [mean_squared_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred)), 
                             mean_absolute_error(y_true, y_pred), r2_score(y_true, y_pred)]
                        )}

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
        f'{prefix}_accuracy': accuracy_score(y_true, y_pred),
        f'{prefix}_balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        f'{prefix}_precision_macro': precision_score(y_true, y_pred, average='macro'),
        f'{prefix}_precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        f'{prefix}_recall_macro': recall_score(y_true, y_pred, average='macro'),
        f'{prefix}_recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        f'{prefix}_f1_macro': f1_score(y_true, y_pred, average='macro'),
        f'{prefix}_f1_weighted': f1_score(y_true, y_pred, average='weighted'),
    }



def concat_for_prediction(X_test, X_train, feature_names, model, consider_abs_diff=False, r1_r2_order=True):
    """
    Creates a dataset where each row represents a comparison between an instance from `X_test`
    and an instance from `X_train`, with the prediction of `model.predict()` as the target value.

    Parameters:
    - X_test (ndarray): Test dataset of shape (n_test_samples, n_features).
    - X_train (ndarray): Train dataset of shape (n_train_samples, n_features).
    - feature_names (list): List of feature names.
    - model: Trained model that has a `.predict()` method.
    - consider_abs_diff (bool): Whether to include absolute differences.
    - r1_r2_order (bool): If True, test instance columns come first; else, train instance columns come first.

    Returns:
    - df_results (pd.DataFrame): DataFrame with pairwise features and model predictions.
    - prediction_matrix (ndarray): Matrix where (i, j) contains model.predict() for X_test[i], X_train[j].
    """
    
    # Convert to DataFrame
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_train = pd.DataFrame(X_train, columns=feature_names)

    # Generate all pairwise combinations (i, j) between test and train sets
    idx_pairs = np.array(list(product(range(len(X_test)), range(len(X_train)))))

    # Get corresponding rows
    row_test = df_test.iloc[idx_pairs[:, 0]].reset_index(drop=True)
    row_train = df_train.iloc[idx_pairs[:, 1]].reset_index(drop=True)

    # Compute absolute differences if required
    if consider_abs_diff:
        df_abs_diff = np.abs(row_test.values - row_train.values)
        df_abs_diff = pd.DataFrame(df_abs_diff, columns=[f"{col}_abs_diff" for col in feature_names])

    # Rename columns for clarity
    row_test.columns = [f"{col}_test" for col in feature_names]
    row_train.columns = [f"{col}_train" for col in feature_names]

    # Order columns based on r1_r2_order flag
    if r1_r2_order:
        df_results = pd.concat([row_test, row_train], axis=1)  # Test columns first
    else:
        df_results = pd.concat([row_train, row_test], axis=1)  # Train columns first

    # Append absolute differences if required
    if consider_abs_diff:
        df_results = pd.concat([df_results, df_abs_diff], axis=1)

    # Insert index pairs
    df_results.insert(0, "indexes", list(map(tuple, idx_pairs)))

    # Prepare feature matrix for prediction
    X_pairwise = df_results.drop(columns=["indexes"]).values

    # Predict using the model
    y_pred = model.predict(X_pairwise)

    # Store predictions
    df_results["prediction"] = y_pred
    
    # Create a prediction matrix
    n_test, n_train = len(X_test), len(X_train)
    prediction_matrix = np.zeros((n_test, n_train))

    # Populate the matrix
    for (i, j), pred in zip(idx_pairs, y_pred):
        prediction_matrix[i, j] = pred

    return df_results, prediction_matrix

def compute_similarity_metrics(vec1, vec2):
    """
    Computes various similarity and distance metrics between two vectors.
    
    Parameters:
    vec1 (np.array): First prediction vector
    vec2 (np.array): Second prediction vector

    Returns:
    dict: Dictionary containing similarity and difference measures
    """
    results = {}

    # Cosine Similarity (Higher is better, 1 means identical)
    results["cosine_similarity"] = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0, 0]

    # Pearson Correlation (Closer to 1 means strong positive correlation)
    results["pearson_correlation"] = np.corrcoef(vec1, vec2)[0, 1]

    # Euclidean Distance (Lower is better)
    results["euclidean_distance"] = euclidean(vec1, vec2)

    # Mean Squared Error (Lower is better)
    results["mse"] = mean_squared_error(vec1, vec2)

    # Structural Similarity Index (SSIM) (Higher is better, close to 1 means similar)
    try:
        results["ssim"] = ssim(vec1, vec2)
    except ValueError:
        results["ssim"] = None  # SSIM may fail for non-normalized data

    # Kolmogorov-Smirnov Test (Lower p-value suggests different distributions)
    ks_stat, ks_p_value = ks_2samp(vec1, vec2)
    results["ks_stat"] = ks_stat
    results["ks_p_value"] = ks_p_value

    return results



def df_pairwise(X, feature_names, dist_matrix, consider_abs_diff=False):
    df_original = pd.DataFrame(X, columns=feature_names)
    indices = df_original.index.to_numpy()

    # Generate all pair combinations (including (i, i))
    idx_pairs = np.array(list(product(indices, repeat=2)))

    # Get row values for both elements in each pair
    row1 = df_original.loc[idx_pairs[:, 0]].reset_index(drop=True)
    row2 = df_original.loc[idx_pairs[:, 1]].reset_index(drop=True)

    # Compute column-wise absolute differences if needed
    if consider_abs_diff:
        df_abs_diff = np.abs(row1.values - row2.values)
        df_abs_diff = pd.DataFrame(df_abs_diff, columns=[f"{col}_abs_diff" for col in feature_names])

    # Compute Euclidean distances using precomputed matrix
    euclidean_distance_sklr = dist_matrix[idx_pairs[:, 0], idx_pairs[:, 1]]

    # Rename columns for distinction
    row1.columns = [f"{col}_1" for col in feature_names]
    row2.columns = [f"{col}_2" for col in feature_names]

    # Merge into a single DataFrame
    df_original_new = pd.concat([row1, row2], axis=1)
    
    # Add distance column
    

    # Append absolute differences if required
    if consider_abs_diff:
        df_original_new = pd.concat([df_original_new, df_abs_diff], axis=1)
        df_original_new["overall_euclidean_distance_sklr"] = euclidean_distance_sklr
    else:
        df_original_new["overall_euclidean_distance_sklr"] = euclidean_distance_sklr

    
    df_original_new.insert(0, "indexes", list(map(tuple, idx_pairs)))

    return df_original_new