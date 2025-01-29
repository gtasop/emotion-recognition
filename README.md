# Dissertation Project

## Overview

This repository contains the code for the dissertation project on emotion recognition. The experiments conducted as part of this research can be replicated using the `main.py` file, where various parameters can be adjusted for each experiment.

## Prerequisites

Before running the project, ensure that you have Python 3.12 installed on your machine. Follow these steps to set up your environment:

1. **Create a virtual environment:**

   Open a terminal and run the following command to create a virtual environment:

   ```bash
   python3.12 -m venv venv
   ```
   
2. **Activate the virtual environment:**
    ```bash
   \venv\Scripts\activate
   ```

3. **Install dependencies**
    ```bash
   pip install -r requirements.txt
   ```

**Note:** The code is set up for parallel execution with CUDA (if available). If your machine supports it, installing CUDA is recommended to train the model faster.

## Data requirements
Due to data privacy concerns, the datasets (/deap & /amigos) folders are included only for the replication of results. After the project has been reviewed the datasets will be deleted from this repository.
###  DEAP folder structure
    └── deap/
        └── data_preprocessed_python/
            ├── s01.dat
            ├── s02.dat
            ├── ...
            └── s32.dat

###  AMIGOS folder structure
    └── amigos/
        └── preprocessed/
            ├── Data_preprocessed_P01/
                ├── Data_preprocessed_P01.mat
            ├── Data_preprocessed_P02/
                ├── Data_preprocessed_P02.mat
            ├── ...
            └── Data_preprocessed_P40/
                └── Data_preprocessed_P40.mat


## How to Run
1. **Running for the first time:** When running the project for the first time, ensure that the configuration variable `generate_images` in `main.py` is set to `True`. This will generate the necessary PSD images for the project.

2. **Execute:**
To execute the code, simply run the `main.py` file:

```bash
python main.py
```
3. **Configuring the parameters to replicate the results:**
The value of each variable in will determine which experiment can be replicated. Below there is a table with the configuration for each experiment. The parameters columns are indicated with (P):

| Model           | Validation type | Classes                                 | Dataset        | (P) model_EEGNet | (P) deap | (P) amigos | (P) cross_subject | (P) original_classes | (P) generate_labels | (P) valence | (P) arousal | (P) num_epochs |
|-----------------|-----------------|-----------------------------------------|----------------|------------------|----------|------------|-------------------|----------------------|---------------------|-------------|-------------|----------------|
| Modified EEGNet | Cross-subject    | Valence                                 | DEAP           | True             | True     | False      | True              | True                 | False               | True        | False       | 500            |
| Modified EEGNet | Cross-subject    | Arousal                                 | DEAP           | True             | True     | False      | True              | True                 | False               | False       | True        | 500            |
| Modified EEGNet | Cross-subject    | Valence/Arousal                         | DEAP           | True             | True     | False      | True              | True                 | False               | True        | True        | 500            |
| Modified EEGNet | Within-subject   | Valence                                 | DEAP           | True             | True     | False      | False             | True                 | False               | True        | False       | 100            |
| Modified EEGNet | Within-subject   | Arousal                                 | DEAP           | True             | True     | False      | False             | True                 | False               | False       | True        | 100            |
| Modified EEGNet | Within-subject   | Valence/Arousal                         | DEAP           | True             | True     | False      | False             | True                 | False               | True        | True        | 100            |
| Modified EEGNet | Cross-subject    | Valence                                 | AMIGOS         | True             | False    | True       | True              | True                 | False               | True        | False       | 200            |
| Modified EEGNet | Cross-subject    | Arousal                                 | AMIGOS         | True             | False    | True       | True              | True                 | False               | False       | True        | 200            |
| Modified EEGNet | Cross-subject    | Valence/Arousal                         | AMIGOS         | True             | False    | True       | True              | True                 | False               | True        | True        | 200            |
| Modified EEGNet | Within-subject   | Valence                                 | AMIGOS         | True             | False    | True       | False             | True                 | False               | True        | False       | 200            |
| Modified EEGNet | Within-subject   | Arousal                                 | AMIGOS         | True             | False    | True       | False             | True                 | False               | False       | True        | 200            |
| Modified EEGNet | Within-subject   | Valence/Arousal                         | AMIGOS         | True             | False    | True       | False             | True                 | False               | True        | True        | 200            |
| Modified EEGNet | Cross-subject    | Valence                                 | DEAP & AMIGOS  | True             | True     | True       | True              | True                 | False               | True        | False       | 100            |
| Modified EEGNet | Cross-subject    | Arousal                                 | DEAP & AMIGOS  | True             | True     | True       | True              | True                 | False               | False       | True        | 100            |
| Modified EEGNet | Cross-subject    | Valence/Arousal                         | DEAP & AMIGOS  | True             | True     | True       | True              | True                 | False               | True        | True        | 100            |
| Modified EEGNet | Within-subject   | Valence                                 | DEAP & AMIGOS  | True             | True     | True       | False             | True                 | False               | True        | False       | 100            |
| Modified EEGNet | Within-subject   | Arousal                                 | DEAP & AMIGOS  | True             | True     | True       | False             | True                 | False               | False       | True        | 100            |
| Modified EEGNet | Within-subject   | Valence/Arousal                         | DEAP & AMIGOS  | True             | True     | True       | False             | True                 | False               | True        | True        | 100            |
| EfficientNet v2 | Cross-subject    | Angry/Happy/Sad/Pleasant/Neutral        | DEAP           | False            | True     | False      | True              | False                | True                | False       | False       | 30             |
| EfficientNet v2 | Cross-subject    | Valence                                 | DEAP           | False            | True     | False      | True              | True                 | True                | True        | False       | 30             |
| EfficientNet v2 | Cross-subject    | Arousal                                 | DEAP           | False            | True     | False      | True              | True                 | True                | False       | True        | 30             |
| EfficientNet v2 | Cross-subject    | Valence/Arousal                         | DEAP           | False            | True     | False      | True              | True                 | True                | True        | True        | 30             |



## Description of Configuration Variables
- `model_EEGNet` : Setting this to `True` will use the EEGNet model. `False` value will use the EfficientNet v2 model
- `deap` : Setting this to `True` will use the DEAP dataset
- `amigos` : Setting this to `True` will use the AMIGOS dataset. This can only be used for the EEGNet model
- `cross_subject`: Setting this to `True` will use cross-subject classification. `False` value will use within-subject classification
- `scheduler`: Setting this to `True` will use the reduction of learning rate on plateau. This can only be used for the EEGNet model
- `original_classes`: This defines if the original classes should be used from the datasets (valence, arousal, dominance, liking). If set to `False` the classes will be relabelled (angry, happy, sad, pleasant, neutral). This can only be used for the EfficientNet v2.
- `generate_images`: Setting this to `True` will generate the PSD plots based on the DEAP dataset
- `generate_labels`: Setting this to `True` will re-generate the labels.csv file which is needed for the different experiments of EfficientNet v2
- `valence`: Setting this to `True` will validate the valence class
- `arousal`: Setting this to `True` will validate the arousal class
- `num_epochs`: Defines the number of epochs that the model will be trained for

## Files description

- `dataset.py`: Contains the dataset classes to read raw data from numpy arrays and combine it with the labels. It also includes functions for splitting the subject data into cross-subject or within-subject batches.
- `dataset_img.py`: Contains the dataset classes to read data from the PSD images folder and combine it with the labels from the .csv file.
- `generate_images.py`: Used to collect data from the DEAP dataset and generate PSD plots for each channel and trial per subject.
- `generate_labels.py`: Used to regenerate the labels.csv file that is utilized by the EfficientNet v2 classifier.
- `load_data.py`: Loads data from the DEAP dataset and converts it into a numpy array.
- `load_data_amigos.py`: Loads data from the AMIGOS dataset, formats it to match DEAP's structure, and converts it into a numpy array.
- `main.py`: The main file used to configure experiment parameters and execute the code.
- `models.py`: Contains the model classes.
- `preprocessing.py`: Contains functions used to preprocess data before training the models. Specifically, it includes functions to segment timesteps into chunks and binarize label values.
- `project_torch.py`: Contains the orchestration steps for the EEGNet model. For example: loading data --> creating data loaders --> initializing the model --> starting training.
- `project_torch_img.py`: Contains the orchestration steps for the EfficientNet v2 model.
- `training.py`: Includes the training, validation, and testing logic for each model.