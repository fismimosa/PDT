#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
# Set environment variables for controlling the number of threads in certain libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from util_exp_funct import * 
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


# Suppress warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from RuleTree import PairwiseDistanceTreeRegressor

import argparse
import pandas as pd

# Add argument parsing
parser = argparse.ArgumentParser(description="Run RANDOM experiments with different random seeds.")
parser.add_argument("--random_state_list", type=int, nargs='+', required=True, help="List of random seeds to use.")
parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to use.")

args = parser.parse_args()

# In[18]:

dataset_name = args.dataset_name
if '.csv' in dataset_name:
    dataset_name = dataset_name.rstrip(".csv")  # Remove .csv if present

try:
    dataframe = pd.read_csv(f'sensitivity_datasets/{dataset_name}.csv')
except FileNotFoundError:
    try:
        dataframe = pd.read_csv(f'large_datasets/{dataset_name}.csv')
    except FileNotFoundError:
        print(f"Error: {dataset_name}.csv not found in both directories.")
        
feature_names = list(dataframe.drop(columns=['label']).columns)


# In[19]:


dataframe


# In[20]:


X = dataframe.drop(columns = ['label']).values
y = np.array(dataframe.label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

num_samples = X_train.shape[0]  


percentages =  [0.2]
sample_sizes = [math.floor(num_samples * p) for p in percentages]

if dataset_name in ['compas','spambase']:
    percentages =  [0.05]
    sample_sizes = [math.floor(num_samples * p) for p in percentages]



fix_feat_dic = {'feat_with_thr' : [True,True]}
absolute_val_dict = {'both_original_and_diff' : [True,True]}


# In[ ]:


for random_seed in args.random_state_list:
    print(f"Running experiment with random_seed = {random_seed}")
    # Your existing experiment logic here
    folder_path = f'TEST_PT_random_results/{dataset_name}'
    file_name = f'random_seed_{random_seed}_{dataset_name}.csv'
    file_path = os.path.join(folder_path, file_name)

    # Skip if the results file already exists
    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Skipping this random seed.")
        continue
        
    np.random.seed(random_seed)  # Fix random seed
    
    all_results_one_seed = []
            
    for perc, sample_size in zip(reversed(percentages), reversed(sample_sizes)):

        if sample_size < 5: #set a base value of 5 if sample size is too low for k-NN with k = 5
            sample_size = 5
        
        results = {'dataset_name': dataset_name, 'random_seed': random_seed, 'perc': perc, 'sample_size': sample_size}
        np.random.seed(random_seed)
        random_indices, selection_time = measure_time(np.random.choice, X_train.shape[0], sample_size, replace=False)
        results.update({f'selection_time_{k}': v for k, v in selection_time.items()})
        
        if perc == 1.0:
            X_train_N, y_train_N = X_train, y_train
            selection_time = {k : 0.0 for k,v in selection_time.items()}
        else:
            X_train_N, y_train_N = X_train[random_indices], y_train[random_indices]
            
        results['sample_size'] = len(y_train_N)
    

        scaler = StandardScaler()

        results.update({f'selection_time_{k}': v for k, v in selection_time.items()})
        
        scaler = StandardScaler()
            
        _, scaler_fit_time = measure_time(scaler.fit, X_train_N)
        results.update({f'scaler_fit_time_{k}': v for k, v in scaler_fit_time.items()})

        X_train_scaled, scaler_transform_train_time = measure_time(scaler.transform, X_train) 
        results.update({f'scaler_transform_train_fulldataset_time_{k}': v for k, v in scaler_transform_train_time.items()})
            
        X_train_N_scaled, scaler_transform_train_time = measure_time(scaler.transform, X_train_N)
        results.update({f'scaler_transform_train_subsample_time_{k}': v for k, v in scaler_transform_train_time.items()})
    
        X_test_scaled, scaler_transform_test_time = measure_time(scaler.transform, X_test)
        results.update({f'scaler_transform_test_time_{k}': v for k, v in scaler_transform_test_time.items()})
            
        dist_matrix_X_train_N_scaled, matrix_train_time = measure_time(pairwise_distances, X_train_N_scaled, metric='euclidean')
        results.update({f'matrix_train_time_{k}': v for k, v in matrix_train_time.items()})
            
        dist_matrix_X_test_scaled, matrix_test_time = measure_time(pairwise_distances, X_test_scaled, metric='euclidean')
        results.update({f'matrix_test_time_{k}': v for k, v in matrix_test_time.items()})


        all_results_different_absolute = []

        for conf_name, value in absolute_val_dict.items():
            conf_results = {'absolute_diff' : conf_name} | results


            feat_to_use = conf_name

            if feat_to_use == 'original_feat':
                df_X_train_N_scaled = df_pairwise(X_train_N_scaled, feature_names, dist_matrix_X_train_N_scaled, consider_abs_diff=False)
                df_X_test_scaled = df_pairwise(X_test_scaled, feature_names, dist_matrix_X_test_scaled,consider_abs_diff=False)
                
            if feat_to_use == 'only_difference':
                df_X_train_N_scaled = df_pairwise(X_train_N_scaled, feature_names, dist_matrix_X_train_N_scaled, consider_abs_diff=True)
                df_X_test_scaled = df_pairwise(X_test_scaled, feature_names, dist_matrix_X_test_scaled, consider_abs_diff=True)
                
                exclude_cols = ['indexes', 'overall_euclidean_distance_sklr']
            
                for df in [df_X_train_N_scaled, df_X_test_scaled]:
                    cols_to_modify = [col for col in df.columns if not col.endswith('_abs_diff') and col not in exclude_cols]
                    df.loc[:, cols_to_modify] = -1
                    
            if feat_to_use == 'both_original_and_diff':
                df_X_train_N_scaled = df_pairwise(X_train_N_scaled, feature_names, dist_matrix_X_train_N_scaled, consider_abs_diff=True)
                df_X_test_scaled = df_pairwise(X_test_scaled, feature_names, dist_matrix_X_test_scaled, consider_abs_diff=True)
    
            
            
            new_feature_names = list(df_X_train_N_scaled.drop(columns=['indexes', 'overall_euclidean_distance_sklr']).columns)
                
            X_train_N_scaled_pairwise = df_X_train_N_scaled[new_feature_names].values        
            y_train_N_scaled_pairwise = df_X_train_N_scaled['overall_euclidean_distance_sklr'].values
                
            X_test_scaled_pairwise = df_X_test_scaled[new_feature_names].values
            y_test_scaled_pairwise = df_X_test_scaled['overall_euclidean_distance_sklr'].values
    
    
            all_model_results_for_perc = []

            for max_depth in [8,16]:
                for k,v in fix_feat_dic.items():
                    fix_feature, fix_threshold = v[0], v[1]
                    
                    params_model = {'max_depth': max_depth, 
                                    'min_samples_leaf': 1,
                                    'min_samples_split': 2,
                                    'random_state' : 42, 
                                    'fix_feature' : fix_feature, 
                                    'fix_threshold' : fix_threshold }
                    
                    
                    model_result = conf_results | params_model
                    
    
                    model = PairwiseDistanceTreeRegressor(**params_model)
                    _, train_time = measure_time(model.fit, X_train_N_scaled, y_train_N_scaled_pairwise, X_train_N_scaled_pairwise)
                    model_result.update({f'PairDistTree_train_time_{k}': v for k, v in train_time.items()})
                
                        # Predictions on training and test sets
                    y_pred_train, predict_train_time = measure_time(model.predict, X_train_N_scaled_pairwise)
                    model_result.update({f'PairDistTree_predict_on_train_time_{k}': v for k, v in predict_train_time.items()})
                
                    y_pred_test, predict_test_time = measure_time(model.predict, X_test_scaled_pairwise)
                    model_result.update({f'PairDistTree_predict_on_test_time_{k}': v for k, v in predict_test_time.items()})
                    
                
                    model_result.update(compute_metrics(y_train_N_scaled_pairwise, y_pred_train, 'PairDistTree_measure_train'))
                    model_result.update(compute_metrics(y_test_scaled_pairwise, y_pred_test, 'PairDistTree_measure_test'))

                    #ADDED IMPORTANCE here you can add feature importance in the computation: we leave this to users decision 
                    #to avoid overfilling results
                    
                    #importance_dict = {k : v for k, v in zip(new_feature_names, model.compute_feature_importances())}
                    #model_result['importance_dict'] = importance_dict

                    X_train_dict = {'full_dataset': (X_train_scaled, y_train)}

                    for X_name, (X_train_curr, y_train_curr) in X_train_dict.items():
                        
                        consider_abs_diff = feat_to_use in ['only_difference', 'both_original_and_diff']

                        model_result['dataset_type'] = X_name 
                        
                        res_r1_r2_test, knn_test_matrix_r1_r2 = concat_for_prediction(
                            X_test_scaled, X_train_curr, feature_names, model, r1_r2_order=True, consider_abs_diff=consider_abs_diff
                        )
                        
                        res_r1_r2_train, knn_train_matrix_r1_r2 = concat_for_prediction(
                            X_train_curr, X_train_curr, feature_names, model, r1_r2_order=True, consider_abs_diff=consider_abs_diff
                        )

                        #different predictions to check for similarity on predictions with swapped order of instances xi-xj/xj-xi
                        #we leave this to users decision to avoid overfilling results

                        #res_r2_r1_test, knn_test_matrix_r2_r1 = concat_for_prediction(
                        #   X_test_scaled, X_train_curr, feature_names, model, r1_r2_order=False, consider_abs_diff=consider_abs_diff
                        #)

                        #res_r2_r1_train, knn_train_matrix_r2_r1 = concat_for_prediction(
                        #    X_train_curr, X_train_curr, feature_names, model, r1_r2_order=False, consider_abs_diff=consider_abs_diff
                        #)

                        
                        #predictions = {
                        #    f"prediction_res_r1_r2_test": np.array(res_r1_r2_test['prediction']),
                        #   f"prediction_res_r2_r1_test": np.array(res_r2_r1_test['prediction']),
                        #   f"prediction_res_r1_r2_train": np.array(res_r1_r2_train['prediction']),
                        #   f"prediction_res_r2_r1_train": np.array(res_r2_r1_train['prediction'])
                        #}
                        
                        #model_result.update(predictions)
                        
                        #similarity_results = {
                        #   "test_similarity": compute_similarity_metrics(predictions['prediction_res_r1_r2_test'], predictions['prediction_res_r2_r1_test']),
                        #    f"train_similarity": compute_similarity_metrics(predictions[f'prediction_res_r1_r2_train'], predictions[f'prediction_res_r2_r1_train'])
                        #}

                        #model_result.update({f'outcome_similarity_on_train_set_{k}': v for k, v in similarity_results[f"test_similarity"].items()})
                        #model_result.update({f'outcome_similarity_on_test_set_{k}': v for k, v in similarity_results[f"train_similarity"].items()})

                        #train_test_matrices = {
                        #        f'train_test_r1_r2_r1_r2': (knn_train_matrix_r1_r2, knn_test_matrix_r1_r2),
                        #       f'train_test_r1_r2_r2_r1': (knn_train_matrix_r1_r2, knn_test_matrix_r2_r1),
                        #        f'train_test_r2_r1_r1_r2': (knn_train_matrix_r2_r1, knn_test_matrix_r1_r2),
                        #       f'train_test_r2_r1_r2_r1': (knn_train_matrix_r2_r1, knn_test_matrix_r2_r1)
                        #    }

                        
                        predictions = {
                            f"prediction_res_r1_r2_test": np.array(res_r1_r2_test['prediction']),
                        }
                        
                        train_test_matrices = {
                                f'train_test_r1_r2_r1_r2': (knn_train_matrix_r1_r2, knn_test_matrix_r1_r2),
                            }
                            
                    
                    # Compute similarity metrics for test and train predictions
                    
                        pt_parameters = {
                                'max_depth': [5],
                                'random_state' : [42],
                                'base_stumps' : [[pt_stump]],
                                'stump_selection' : ['best'],
                                'prune_useless_leaves' : [False],   
                            }
                        
                        for config_possible in ParameterGrid(pt_parameters):
                            
                            for matrix_name, (train_matrix, test_matrix) in train_test_matrices.items():
                                if matrix_name != 'train_test_r1_r2_r1_r2':
                                    continue
                                
                                # Initialize model with parameters
                                def custom_measure_feature(x1, x2):
                                    
                                    combined_array = np.concatenate((x1, x2), axis=0)
                                    
                                    combined_array = combined_array.reshape(1, -1)

                                    return model.predict(combined_array)[0]
                                    
                                def custom_measure_abs_diff(x1, x2):
                                    abs_diff = np.abs(x1 - x2)
                                   
                                    combined_array = np.concatenate((x1, x2, abs_diff), axis=0)
                                    
                                    combined_array = combined_array.reshape(1, -1)

                                   # print(model.predict(combined_array))
                                    
                                    return model.predict(combined_array)[0]

                                pt_model = RuleTreeClassifier(**config_possible, distance_measure = custom_measure_abs_diff, 
                                                             )
                                
                                
                        
                                _, train_time = measure_time(pt_model.fit, X_train_curr, y_train_curr)
                               # print(pt_model.root.stump.feature_original)
                               # print(pt_model.root.stump.threshold_original)
                                
                          
                                model_result.update({f'PT_train_time_{k}': v for k, v in train_time.items()})
                            
                                # Store model results
                                pt_config = {'pt_' + k :v for k, v in config_possible.items()}
                                
                                model_result =  model_result | pt_config.copy()
                                
                                model_result[f'train_test_case'] = matrix_name

                                y_pred_train, predict_train_time = measure_time(pt_model.predict, X_train_curr)
                                model_result.update({f'PT_predict_on_train_time_{k}': v for k, v in predict_train_time.items()})
                                
                                y_pred_test, predict_test_time = measure_time(pt_model.predict, X_test_scaled)
                                model_result.update({f'PT_predict_on_test_time_{k}': v for k, v in predict_test_time.items()})

            
                                # Add additional metrics if required
                                # model_result.update(compute_metrics(y_true, y_pred, prefix=matrix_name))
                        
                                # Print or store model results
                                #print(f"Config: {config_possible}, Case: {matrix_name}")
                                model_result.update(compute_metrics_classifier(y_train_curr, y_pred_train, f'PT_measure_train'))
                                model_result.update(compute_metrics_classifier(y_test, y_pred_test, f'PT_measure_test'))
                                
                                
                                all_model_results_for_perc.append(model_result)
                    
                    a,b = fix_feature, fix_threshold
                    print(f'{dataset_name} Computed for seed {random_seed} with percentage {perc} and depth {max_depth} {a} {b} {conf_name}')
                    
                    
                    #all_model_results_for_perc.append(model_result)
                    
            all_results_different_absolute.append(all_model_results_for_perc)
            print(f'{dataset_name} Computed for seed {random_seed} with percentage {perc} and depth {max_depth} {a} {b} {conf_name}')
            
        all_results_one_seed.append(all_results_different_absolute)
        
    all_results_one_seed = sum(all_results_one_seed,[])
    all_results_one_seed = sum(all_results_one_seed,[])
    df = pd.DataFrame(all_results_one_seed)
    #display(pd.DataFrame(all_results_one_seed))

    results_df = df
    folder_path = f'TEST_PT_random_results/{dataset_name}'  # Corrected variable name
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist
    
    file_name = f'random_seed_{random_seed}_{dataset_name}.csv'
    results_df.to_csv(os.path.join(folder_path, file_name), index=False)   
                        
                        

        


# In[ ]:





# In[ ]:





# In[ ]:




