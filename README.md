# PairwiseDistanceTree
Instance-based models offer natural interpretability by making decisions based on concrete examples. 
However, their transparency is often hindered by complex similarity measures, which are difficult to interpret, especially in high-dimensional datasets. 
To address this issue, this paper presents a meta-learning framework that enhances the interpretability of instance-based models by replacing traditional, complex pairwise distance functions with interpretable pairwise distance trees. 
These trees prioritize simplicity and transparency while preserving the model’s effectiveness. 
By offering a clear decision-making process, the framework makes the instance selection more understandable.
Also, the framework mitigates the computational burden of instance-based models, which typically require calculating all pairwise distances. 
The method significantly reduces computational complexity by leveraging the generalization capabilities of pairwise distance trees and employing sampling strategies to select representative subsets. 
Our experiments demonstrate that the proposed approach improves computational efficiency with only a modest trade-off in accuracy while substantially enhancing the interpretability of the learned distance measure.

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
from pdt.model import PairwiseDT_KNN

dt = PairwiseDT_KNN()
x = ...
y = ...

# PairwiseDT_KNN follows a similar sklearn-like training interface, with max_depth, n_neighbors, etc. as available parameters
dt.fit(x, y, max_depth=4)
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
