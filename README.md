# Facial Landmark Detection

## Requirements

```
python3.8
```

## Installation

```
pip install -r requirements.txt
```

## data path

```
project
│   README.md
|   requirements.txt
│   main.py
|   train.py
|   myDataset.py
|   myModel.py
|   best_mnasnet0_75_3.pt
│
└───data
    │
    └───aflw_test
    |   │   image00002.jpg
    |   │   image00004.jpg
    |   |   ...
    |
    └───aflw_val
    |   |   annot.pkl
    |   │   image00013.jpg
    |   │   image00036.jpg
    |   |   ...
    |
    └───synthetics_train
        |   annot.pkl
        │   000000.jpg
        │   000001.jpg
        |   ...
```

## Run

default checkpoint path is best.pt

default data path is ./data

### training

```
python main.py -m train -c [checkpoint_path] -d [data_path]
```

### testing

```
python main.py -m test -c [checkpoint_path] -d [data_path]
```

It will create a solution.txt file

Note: the model path which can generate the score on Leaderboard is "best_mnasnet0_75_3.pt"
