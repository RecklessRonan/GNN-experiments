Software Versions:
Python: 3.7.8
numpy: 1.18.5
scipy: 1.5.3
pytorch: 1.6.0
networkx: 2.5
scikit-learn: 0.23.2
dgl: it is only used to run baseline GeomGCN. If you need to run this baseline, you need to use this version of dgl: 0.4.3
If you do not need to run GeomGCN baseline, you can install any version of dgl or remove the the codes related to the GeomGCN baseline (in process.py and full-supervised.py).
To replicate the results of Table 1, you can using ./table_1_[model_name].sh to obatin the results of the specified model.
To replicate Table 2, you can still use hyparameters used in table_1_[model_name].sh and modify the layers.
To replicate Table B1, you can run ./table_B1.sh

