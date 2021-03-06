python3 train.py --dataset_name cornell --lr 0.1 --weight_decay 0.0001 --dropout 0 --hidden 64 --layers 2 --fixed_splits 1 --variant
python3 train.py --dataset_name wisconsin --lr 0.01 --weight_decay 0.00005 --dropout 0.1 --hidden 64 --layers 2 --fixed_splits 1 --variant
python3 train.py --dataset_name texas --lr 0.01 --weight_decay 0.001 --dropout 0.1 --hidden 64 --layers 2 --fixed_splits 1 --variant
python3 train.py --dataset_name film --lr 0.01 --weight_decay 0.005 --dropout 0 --hidden 64 --layers 2 --fixed_splits 1 --variant
python3 train.py --dataset_name chameleon --lr 0.05 --weight_decay 0.000005 --dropout 0.8 --hidden 64 --layers 2 --fixed_splits 1 --variant
python3 train.py --dataset_name squirrel --lr 0.05 --weight_decay 0.000005 --dropout 0.7 --hidden 64 --layers 2 --fixed_splits 1 --variant
python3 train.py --dataset_name cora --lr 0.01 --weight_decay 0.0001 --dropout 0.6 --hidden 64 --layers 2 --fixed_splits 1 --variant
python3 train.py --dataset_name citeseer --lr 0.01 --weight_decay 0.00005 --dropout 0.5 --hidden 64 --layers 2 --fixed_splits 1 --variant
python3 train.py --dataset_name pubmed --lr 0.01 --weight_decay 0.0001 --dropout 0.3 --hidden 64 --layers 2 --fixed_splits 1 --variant