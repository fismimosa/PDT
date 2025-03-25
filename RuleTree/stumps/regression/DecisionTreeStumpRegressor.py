import copy
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_poisson_deviance
from sklearn.tree import DecisionTreeRegressor

from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump
from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier

from RuleTree.utils.data_utils import get_info_gain, _get_info_gain


class DecisionTreeStumpRegressor(DecisionTreeRegressor, RuleTreeBaseStump):
    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        return DecisionTreeStumpClassifier.get_rule(self,
                                                    columns_names=columns_names,
                                                    scaler=scaler,
                                                    float_precision=float_precision)

    def node_to_dict(self):
        rule = self.get_rule(float_precision=None)

        rule["stump_type"] = self.__class__.__name__
        rule["samples"] = self.tree_.n_node_samples[0]
        rule["impurity"] = self.tree_.impurity[0]

        rule["args"] = {
                           "unique_val_enum": self.unique_val_enum,
                       } | self.kwargs

        rule["split"] = {
            "args": {}
        }

        return rule

    def dict_to_node(self, node_dict, X=None):
        assert 'feature_idx' in node_dict
        assert 'threshold' in node_dict
        assert 'is_categorical' in node_dict

        self.feature_original = np.zeros(3)
        self.threshold_original = np.zeros(3)

        self.feature_original[0] = node_dict["feature_idx"]
        self.threshold_original[0] = node_dict["threshold"]
        self.is_categorical = node_dict["is_categorical"]

        args = copy.deepcopy(node_dict.get("args", dict()))
        self.unique_val_enum = args.pop("unique_val_enum", np.nan)
        self.kwargs = args

        self.__set_impurity_fun(args["criterion"])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_categorical = False
        self.kwargs = kwargs
        self.unique_val_enum = None
        self.threshold_original = None
        self.feature_original = None

        self.impurity_fun = kwargs['criterion'] if 'criterion' in kwargs else "squared_error"

    @classmethod
    def _get_impurity_fun(cls, imp):
        if imp == "squared_error":
            return mean_squared_error
        elif imp == "friedman_mse":
            raise Exception("not implemented") # TODO: implement
        elif imp == "absolute_error":
            return mean_absolute_error
        elif imp == "poisson":
            return mean_poisson_deviance
        else:
            return imp


    @classmethod
    def _impurity_fun(cls, impurity_fun, **x):
        f = cls._get_impurity_fun(impurity_fun)
        return f(**x) if len(x["y_true"]) > 0 else 0 # TODO: check

    def get_params(self, deep=True):
        return self.kwargs

    def fit(self, X, y, previous_feature, previous_threshold , idx=None, context=None, sample_weight=None, check_input=True,
            
            
            first_pair_feats = None ,
            second_pair_feats = None,
            abs_diff_feats = None,  
            n_original_features = None
            ):
        
        if idx is None:
            idx = slice(None)
        X = X[idx]
        y = y[idx]

        self.feature_analysis(X, y)
        best_info_gain = -float('inf')
        
        
 
        if len(self.numerical) > 0:

            if previous_feature is not None and previous_feature not in abs_diff_feats:
                #print('PRINTING WITH PREVIOUS FEEATURE')
                if previous_feature in first_pair_feats:
                  #  print('FIRST GROUPS')
                  #  print(previous_feature)
                    feat_choice = self.numerical[previous_feature + n_original_features]  # Ensure valid index
                  #  print(feat_choice)
                if previous_feature in second_pair_feats:
                    #print('SECOND GROUP')
                    #print(previous_feature)
                    feat_choice = self.numerical[previous_feature - n_original_features]  # Ensure valid index
                   # print(feat_choice)
                    
        
               
               
                if previous_threshold is not None:
                    #print('PRINTING WITH PREVIOUS THRESHOLD')
                    len_x = len(X)
                    
                    #X_split = X[:, feat_choice] <= previous_threshold  # Use fixed feature and threshold
                    X_split = X[:, feat_choice:feat_choice+1] <= previous_threshold 
                    #X[:, i:i+1]
                    len_left = np.sum(X_split)
                    curr_pred = np.ones((len(y), )) * np.mean(y)
                    
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        l_pred = np.ones((len(y[X_split[:, 0]]),)) * np.mean(y[X_split[:, 0]])
                        r_pred = np.ones((len(y[~X_split[:, 0]]),)) * np.mean(y[~X_split[:, 0]])

                        info_gain = _get_info_gain(self._impurity_fun(self.impurity_fun, y_true=y, y_pred=curr_pred),
                                                   self._impurity_fun(self.impurity_fun, y_true=y[X_split[:, 0]], y_pred=l_pred),
                                                   self._impurity_fun(self.impurity_fun, y_true=y[~X_split[:, 0]], y_pred=r_pred),
                                                   len_x,
                                                   len_left,
                                                   len_x - len_left)
                        
                    self.feature_original = [feat_choice, -2, -2]
                    self.threshold_original = np.array([previous_threshold, -2, -2])
                    self.n_node_samples = [len_x,len_left, len_x - len_left ]

                    imp_parent = self._impurity_fun(self.impurity_fun, y_true=y, y_pred=curr_pred)
                    imp_child_l = self._impurity_fun(self.impurity_fun, y_true=y[X_split[:, 0]], y_pred=l_pred)
                    imp_child_r = self._impurity_fun(self.impurity_fun, y_true=y[~X_split[:, 0]], y_pred=r_pred)
                    
                    self.fix_thr_tree_ = {}
                    self.fix_thr_tree_['impurity'] = [imp_parent, imp_child_l, imp_child_r]
                    self.fix_thr_tree_['n_node_samples'] = [len_x, len_left,len_x - len_left]
                    
                    #self.tree_.impurity = 
                


                    
                else:
                    mask = np.zeros(X.shape[1], dtype=bool)  # Initialize mask with False
                    mask[feat_choice] = True  # Keep only the selected feature
                        
                    X_current = np.where(mask, X, -1)  # Apply mask: keep selected features, zero out others
                    
                    super().fit(X_current[:, self.numerical], y, sample_weight=sample_weight, check_input=check_input)
                    self.feature_original = [self.numerical[x] if x != -2 else x for x in self.tree_.feature]
                    self.threshold_original = self.tree_.threshold
                    self.n_node_samples = self.tree_.n_node_samples
                    
                    best_info_gain = get_info_gain(self)
                
                
            
            else:
                super().fit(X[:, self.numerical], y, sample_weight=sample_weight, check_input=check_input)
                self.feature_original = [self.numerical[x] if x != -2 else x for x in self.tree_.feature]
                self.threshold_original = self.tree_.threshold
                self.n_node_samples = self.tree_.n_node_samples
                best_info_gain = get_info_gain(self)
                
            
            
        self._fit_cat(X, y, best_info_gain)

        return self

    def _fit_cat(self, X, y, best_info_gain):
        if self.max_depth > 1:
            raise Exception("not implemented") # TODO: implement?

        len_x = len(X)

        if len(self.categorical) > 0 and best_info_gain != float('inf'):
            for i in self.categorical:
                for value in np.unique(X[:, i]):
                    X_split = X[:, i:i+1] == value
                    len_left = np.sum(X_split)
                    curr_pred = np.ones((len(y), ))*np.mean(y)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        l_pred = np.ones((len(y[X_split[:, 0]]),)) * np.mean(y[X_split[:, 0]])
                        r_pred = np.ones((len(y[~X_split[:, 0]]),)) * np.mean(y[~X_split[:, 0]])

                        info_gain = _get_info_gain(self._impurity_fun(self.impurity_fun, y_true=y, y_pred=curr_pred),
                                                   self._impurity_fun(self.impurity_fun, y_true=y[X_split[:, 0]], y_pred=l_pred),
                                                   self._impurity_fun(self.impurity_fun, y_true=y[~X_split[:, 0]], y_pred=r_pred),
                                                   len_x,
                                                   len_left,
                                                   len_x - len_left)

                    if info_gain > best_info_gain:
                        best_info_gain = info_gain
                        self.feature_original = [i, -2, -2]
                        self.threshold_original = np.array([value, -2, -2])
                        self.unique_val_enum = np.unique(X[:, i])
                        self.is_categorical = True


    def apply(self, X, check_input=False):
        if len(self.feature_original) < 3:
            return np.ones(X.shape[0])

        if not self.is_categorical:
            y_pred = np.ones(X.shape[0], dtype=int) * 2
            X_feature = X[:, self.feature_original[0]]
            y_pred[X_feature <= self.threshold_original[0]] = 1
            
            return y_pred
        else:
            y_pred = np.ones(X.shape[0], dtype=int) * 2
            X_feature = X[:, self.feature_original[0]]
            y_pred[X_feature == self.threshold_original[0]] = 1

            return y_pred

