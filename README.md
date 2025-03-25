# PairwiseDistanceTree
Instance-based models offer natural interpretability by making decisions based on concrete examples. 
However, their transparency is often hindered by complex similarity measures, which are difficult to interpret, especially in high-dimensional datasets. 
To address this issue, this paper presents a meta-learning framework that enhances the interpretability of instance-based models by replacing traditional, complex pairwise distance functions with interpretable pairwise distance trees. 
These trees prioritize simplicity and transparency while preserving the model’s effectiveness. 
By offering a clear decision-making process, the framework makes the instance selection more understandable.
Also, the framework mitigates the computational burden of instance-based models, which typically require calculating all pairwise distances. 
The method significantly reduces computational complexity by leveraging the generalization capabilities of pairwise distance trees and employing sampling strategies to select representative subsets. 
Our experiments demonstrate that the proposed approach improves computational efficiency with only a modest trade-off in accuracy while substantially enhancing the interpretability of the learned distance measure.

![PairTree Overview](https://github.com/user-attachments/assets/b944e9d7-bfef-4c1a-92db-82e3d603473c)
![](path_to_image)
*At inference time, given a query instance $`x`$, the model selects relevant neighbors from the memory $`\langle X, Y \rangle`$ by evaluating $`r(x,x_i)`$ and applies an inference policy $`\phi`$, i.e., majority voting, to produce a final prediction. Each prediction can be inspected, since the distance function employed for neighborhood selection is fully interpretable.*

# Quickstart
## Installation
Installation through git:
```shell
git clone https://github.com/unknown/PDT
mkvirtualenv -p python3.12 PDT  # optional, creates virtual environment

cd PDT
pip install -r src/requirements.txt
```
or directly through `pip`:
```shell
pip install pdt #coming soon
```

## Training trees
PDT follows the classic sklearn `fit`/`predict` interface.  

```python
from RuleTree import PairwiseDistanceTreeRegressor
from util_exp_funct import df_pairwise

#assume list of feature names
feature_names

#assume train dataset
X_train, y_train 

#get pairwise matrix
X_train_matrix = pairwise_distances(X_train) 

#get dataset of pairs with distance as target
df_X_train = df_pairwise(X_train,
                        feature_names,
                        X_train_matrix,
                        consider_abs_diff=True)


new_feature_names = list(df_X_train.drop(columns=['indexes', 'overall_euclidean_distance_sklr']).columns)

X_train_pairwise = df_X_train[new_feature_names].values

dt = PairwiseDistanceTreeRegressor()


# PairwiseDT_KNN follows a similar sklearn-like training interface, with max_depth, n_neighbors, etc. as available parameters
#learn pairwise distance approximator

dt.fit(X_train, y_train, X_pairwise)
```


## Using learned distance for classifiers

```python
from util_exp_funct import concat_for_prediction

 #learn pairwise distance approximator
dt.fit(X_train, y_train, X_pairwise)

#create distance matrix with approxiamated distances
_, test_matrix = concat_for_prediction(X_test, X_train, feature_names, dt, consider_abs_diff=True)
_, train_matrix = concat_for_prediction(X_train, X_train, feature_names, dt, consider_abs_diff=True)


knn_model = KNeighborsClassifier(n_neighbors=3, metric='precomputed')

#fit case-based model on computed approximated distance matrix
knn_model.fit(train_matrix , y_train)

#predict
y_pred_test = knn_model.predict(test_matrix)
```


## Docs and reference

You can cite this work with
```
@inproceedings{abc,
  author       = {unk},
  title        = {Interpretable Instance-Based Learning through Pairwise Distance Trees},
  booktitle    = {},
  series       = {},
  volume       = {},
  pages        = {},
  publisher    = {},
  year         = {}
}
```
