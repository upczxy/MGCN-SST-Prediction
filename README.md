# Sea-Surface-Temperature-Prediction-with-Memory-Graph-Convolutional-Networks
## Introduction 
We propose a memory graph convolutional network for SST prediction prediction. The method is a spatiotemporal prediction method for SST sequence.
## Prerequisites
Our code is based on Python3 (>= 3.6). The major libraries are listed as follows:
* TensorFlow (>= 1.9.0)
* NumPy (>= 1.15)
* SciPy (>= 1.1.0)
* Pandas (>= 0.23)
## Folder structure
```
├── data_loader
│   ├── data_utils.py
│   └── __init__.py
├── dataset
│   ├── sstmonth.csv
│   └── sstw2.csv
├── main.py
├── models
│   ├── base_model.py
│   ├── __init__.py
│   ├── layers.py
│   ├── tester.py
│   └── trainer.py
├── output
│   ├── models
│   └── tensorboard
├── README.md
└── utils
    ├── __init__.py
    ├── math_graph.py
    └── math_utils.py
```
## Data
The datasets is as following:  
- sstmonth.csv : Historical monthly mean SST values with shape of [len * grid point].
- sstw2.csv : Weighted Adjacency Matrix with shape of [grid point* grid point].

## Training
python main.py --n_route {%grid point%} --graph {%weight_matrix.csv%} --n_his {%6%} --n_pred {%3%}  

