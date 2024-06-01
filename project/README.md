# Advanced topics in computer vision

## Abstract

This project aims to perform the following: Given a photo from the webcam, move the cursor where the user points their eyes.

## Prerequisites

This project requires:
1. A functional distribution of Python 3.11+. Below, `python` is assumed to be Python 3.
1. The dataset in the `./dataset/original` directory, containing:
    1. The `train.csv`, `validation.csv`, `test.csv` CSV files
    1. The `images/` directory, with the photos files from the CSV file above 


## Project setup

Run the following commands:

```bash
$ python -m venv venv
$ source ./venv/bin/activate
$ pip install -r requirements.txt
$ deactivate
```

After these commands are run, run the command below before running the project:

```bash
$ source ./venv/bin/activate
```

## Running

### Data preprocessing

Since `main.py` assumes that the data is already preprocessed, use the following command:

```bash
$ python ./src/preprocess.py
```

This will create a folder in `./dataset/64x64` (default value) with all the preprocessed data.

### Training

Run:

```bash
$ python ./src/train.py
```
