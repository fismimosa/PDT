import heapq
import warnings

import numpy as np
import sklearn
from sklearn import tree
from sklearn.base import RegressorMixin

from RuleTree.stumps.regression.PairwiseDistanceDecisionTreeStumpRegressor import PairwiseDistanceDecisionTreeStumpRegressor
from RuleTree.tree.RuleTree import RuleTree
from RuleTree.tree.RuleTreeNode import RuleTreeNode
from RuleTree.utils.data_utils import get_info_gain


class PairwiseDistanceTreeRegressor(RuleTree, RegressorMixin):
    def __init__(self,
                 max_leaf_nodes=float('inf'),
                 min_samples_split=2,
                 max_depth=float('inf'),
                 prune_useless_leaves=False,
                 base_stumps: RegressorMixin | list = None,
                 stump_selection:str='random',
                 random_state=None,

                 criterion='squared_error',
                 splitter='best',
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features=None,
                 min_impurity_decrease=0.0,
                 ccp_alpha=0.0,
                 monotonic_cst=None,
                 oblique = False,
                 oblique_params = {},
                 oblique_split_type =  'householder',
                 force_oblique = False,
                 
                 fix_feature = False,
                 fix_threshold = False
                 ):
        
        if base_stumps is None:
            base_stumps = PairwiseDistanceDecisionTreeStumpRegressor(
                max_depth=1,
                criterion=criterion,
                splitter=splitter,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                random_state=random_state,
                min_impurity_decrease=min_impurity_decrease,
                ccp_alpha=ccp_alpha,
                monotonic_cst=monotonic_cst
            )

        super().__init__(max_leaf_nodes=max_leaf_nodes,
                         min_samples_split=min_samples_split,
                         max_depth=max_depth,
                         prune_useless_leaves=prune_useless_leaves,
                         base_stumps=base_stumps,
                         stump_selection=stump_selection,
                         random_state=random_state)

        self.criterion = criterion
        self.splitter = splitter
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst
        self.oblique = oblique
        self.oblique_params = oblique_params
        self.oblique_split_type = oblique_split_type
        self.force_oblique = force_oblique
        
        self.fix_feature = fix_feature
        self.fix_threshold = fix_threshold

    def is_split_useless(self, X, clf: tree, idx: np.ndarray):
        labels = clf.apply(X[idx])
        return len(np.unique(labels)) == 1

    def queue_push(self, node: RuleTreeNode, idx: np.ndarray):
        heapq.heappush(self.queue, (len(node.node_id), next(self.tiebreaker), idx, node))
        
        
    def fit(self, X: np.array, y: np.array = None, X_pairwise = None, **kwargs):
        
        self.n_original_features = X.shape[1]
       # print(self.n_original_features)

        if X_pairwise is not None:
            X = X_pairwise
            self.first_pair_feats = [i for i in range(self.n_original_features)]
            self.second_pair_feats = [i + self.n_original_features for i in range(self.n_original_features)]
            self.abs_diff_feats = [i + self.n_original_features for i in self.second_pair_feats]
            
            
            
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features = X.shape[1]
        self._set_stump()

        idx = np.arange(X.shape[0])

        self.root = self.prepare_node(y, idx, "R")
        
        self.root.fix_feature = False
    
        self.queue_push(self.root, idx)

        nbr_curr_nodes = 0
        while len(self.queue) > 0 and nbr_curr_nodes + len(self.queue) < self.max_leaf_nodes:
            idx, current_node = self.queue_pop()
            
            
            if len(idx) < self.min_samples_split:
                self.make_leaf(current_node)
                current_node.medoids_index = self.compute_medoids(X, y, idx=idx, **kwargs)
                nbr_curr_nodes += 1
                continue

            if nbr_curr_nodes + len(self.queue) + 1 >= self.max_leaf_nodes:
                self.make_leaf(current_node)
                current_node.medoids_index = self.compute_medoids(X, y, idx=idx, **kwargs)
                nbr_curr_nodes += 1
                continue

            if self.max_depth is not None and current_node.get_depth() >= self.max_depth:
                self.make_leaf(current_node)
                current_node.medoids_index = self.compute_medoids(X, y, idx=idx, **kwargs)
                nbr_curr_nodes += 1
                continue

            if self.check_additional_halting_condition(y=y, curr_idx=idx):
                self.make_leaf(current_node)
                current_node.medoids_index = self.compute_medoids(X, y, idx=idx, **kwargs)
                nbr_curr_nodes += 1
                continue
    
            if current_node.parent is None:
                # Root node: Perform a free split
                clf = self.make_split(X, y, idx=idx, previous_feature=None, previous_threshold=None, **kwargs)
                
                # Next level should keep the feature (and possibly threshold) fixed
                #node_l_fix_feat_condition, node_r_fix_feat_condition = True, True  
                current_node.fix_feature = True

            elif not self.fix_feature:
                clf = self.make_split(X, y, idx=idx, previous_feature=None, previous_threshold=None, **kwargs)
                
            elif self.fix_feature:
                parent_feature = current_node.parent.stump.feature_original[0]
                parent_threshold = current_node.parent.stump.threshold_original[0]

                if parent_feature in self.abs_diff_feats: #if i have chosen an abs diff featrue, perform a free split
                    clf = self.make_split(X, y, idx=idx, previous_feature=None, previous_threshold=None, **kwargs)
                    current_node.fix_feature = True

                #if the previous split was feature based then check the force condition
                elif current_node.parent.fix_feature:
                    # If the parent was not abs diff and enforced a feature, maintain the same feature for this node
                    if self.fix_threshold:
                        #maintain the same threshold
                        clf = self.make_split(X, y, idx=idx, previous_feature=parent_feature, previous_threshold=parent_threshold, **kwargs)
                    else:
                        clf = self.make_split(X, y, idx=idx, previous_feature=parent_feature, previous_threshold=None, **kwargs)
                    
                    # Next level should be free
                    #node_l_fix_feat_condition, node_r_fix_feat_condition = False, False
                    current_node.fix_feature = False
                    
                
                elif not current_node.parent.fix_feature:
                    # If the parent did not enforce a feature, perform a free split
                    clf = self.make_split(X, y, idx=idx, previous_feature=None, previous_threshold=None, **kwargs)
                    
                    # Next level should keep the feature fixed
                    #node_l_fix_feat_condition, node_r_fix_feat_condition = True, True
                    current_node.fix_feature = True

            
            labels = clf.apply(X[idx])
            

           

            
           
            name_clf = clf.__class__.__module__.split('.')[-1]
            #print(name_clf)
            
            if name_clf in ['ObliqueDecisionTreeStumpClassifier',
                            'DecisionTreeStumpClassifier']:
                current_node.medoids_index = self.compute_medoids(X, y, idx=idx, **kwargs)
                
                
            global_labels = clf.apply(X)
            current_node.balance_score_global = (np.min(np.unique(global_labels, return_counts= True)[1]) / global_labels.shape[0])
            current_node.balance_score = current_node.balance_score_global

            if self.is_split_useless(X=X, clf=clf, idx=idx):
                self.make_leaf(current_node)
                current_node.medoids_index = self.compute_medoids(X, y, idx=idx, **kwargs)
                nbr_curr_nodes += 1
                continue

            idx_l, idx_r = idx[labels == 1], idx[labels == 2]

            current_node.set_stump(clf)
            current_node.node_l = self.prepare_node(y, idx_l, current_node.node_id + "l", )
            current_node.node_r = self.prepare_node(y, idx_r, current_node.node_id + "r", )
            
            #added here
            
          

            
            
            current_node.node_l.parent, current_node.node_r.parent = current_node, current_node
            
            
            
            #current_node.balance_score = (np.min(np.unique(labels, return_counts= True)[1]) / labels.shape[0])
            
            #global_labels = clf.apply(X)
            #current_node.balance_score_global = (np.min(np.unique(global_labels, return_counts= True)[1]) / global_labels.shape[0])
           
            #current_node.balance_score = (np.min(np.unique(global_labels, return_counts= True)[1]) / global_labels.shape[0])
           
            
            

            self.queue_push(current_node.node_l, idx_l)
            self.queue_push(current_node.node_r, idx_r)

        if self.prune_useless_leaves:
            self.root = self.root.simplify()

        self._post_fit_fix()

        return self

    def make_split(self, X: np.ndarray, y, idx: np.ndarray, previous_feature, previous_threshold, **kwargs) -> tree:
        if self.stump_selection == 'random':
            stump = self._get_random_stump(X)
            stump.fit(X=X,
                      y=y,
                      idx=idx,
                      context=self,
                      
                      previous_feature = previous_feature,
                      previous_threshold = previous_threshold,
                      
                      first_pair_feats = self.first_pair_feats , #addded here
                      second_pair_feats = self.second_pair_feats,
                      abs_diff_feats = self.abs_diff_feats,  
                      n_original_features = self.n_original_features,
                      
                      **kwargs)
        elif self.stump_selection == 'best':
            clfs = []
            info_gains = []
            for _, stump in self._filter_types(X):
                stump = sklearn.clone(stump)
                stump.fit(X=X,
                          y=y,
                          idx=idx,
                          context=self,
                          
                          previous_feature = previous_feature,
                          previous_threshold = previous_threshold,
                          
                          
                          first_pair_feats = self.first_pair_feats , #added here
                          second_pair_feats = self.second_pair_feats,
                          abs_diff_feats = self.abs_diff_feats,  
                          
                          **kwargs)

                gain = get_info_gain(stump)
                info_gains.append(gain)
                
                clfs.append(stump)

            stump = clfs[np.argmax(info_gains)]
        else:
            raise TypeError('Unknown stump selection method')

        return stump

   
    def compute_medoids(self, X: np.ndarray, y, idx: np.ndarray, **kwargs):
        pass
        
    def prepare_node(self, y: np.ndarray, idx: np.ndarray, node_id: str) -> RuleTreeNode:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prediction = float(np.mean(y[idx]))
            prediction_std = float(np.std(y[idx]))

        return RuleTreeNode(
            node_id=node_id,
            prediction=prediction,
            prediction_probability=prediction_std,
            parent=None,
            stump=None,
            node_l=None,
            node_r=None,
            samples=len(y[idx]),
            classes=self.classes_
        )

    def _get_stumps_base_class(self):
        return RegressorMixin
        
    def _get_prediction_probas(self, current_node = None, probas=None):
        if probas is None:
            probas = []
            
        if current_node is None:
            current_node = self.root
        
    
        if current_node.prediction is not None:
            probas.append(current_node.prediction)
           
        if current_node.node_l:
            self._get_prediction_probas(current_node.node_l, probas)
            self._get_prediction_probas(current_node.node_r, probas)
        
        return probas
    
    
    def local_interpretation(self, X, joint_contribution = False):
        leaves, paths, leaf_to_path, values = super().local_interpretation(X = X,
                                                                           joint_contribution = joint_contribution)
        
        values = values.squeeze(axis=1)
        biases = np.full(X.shape[0], values[paths[0][0]])
        line_shape = X.shape[1]
        
        return super().eval_contributions(
                                        leaves=leaves,
                                        paths=paths,
                                        leaf_to_path=leaf_to_path,
                                        values=values,
                                        biases=biases,
                                        line_shape=line_shape,
                                        joint_contribution=joint_contribution
                                    )

