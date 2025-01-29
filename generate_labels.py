import os
import pandas as pd
import numpy as np
from load_data import load_data_deap
from preprocessing import binarize_array

def generate_labels(trial_index, channel_name, labels_list, label, label2=None):
    img_name = f'channel_{channel_name}_psd.png'
    img_path = os.path.join(trial_index, img_name)
    if label2 is None:
        labels_list.append({'trial_name': img_path, 'label1': label})
    else:
        labels_list.append({'trial_name': img_path, 'label1': label, 'label2': label2})


def generate_new_labels(original_classes=True, valence=True, arousal=False):
    data, labels = load_data_deap(nomarlize_labels=False)
    channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
                'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
    num_classes = 1
    labels = labels[:, :, :2]

    if original_classes:
        labels = binarize_array(labels, threshold=5.0)
        # Select just valence
        if valence and not (arousal):
            labels = labels[:, :, 0]
            num_classes = 1
        # Select just arousal
        elif not (valence) and arousal:
            labels = labels[:, :, 1]
            num_classes = 1
        # Select valence and arousal
        else:
            labels = labels[:, :, :2]
            num_classes = 2
        data = data.reshape(-1, data.shape[2], data.shape[3])
        if num_classes == 1:
            labels = labels.reshape(-1)
        else:
            labels = labels.reshape(-1, labels.shape[2])
    else:
        data = data.reshape(-1, data.shape[2], data.shape[3])
        labels = labels.reshape(-1, labels.shape[2])

        valence_thresholds = [4.5, 5.5]
        arousal_thresholds = [4.5, 5.5]
        # Initialize the array to hold the new 5 classes
        classes = np.empty((labels.shape[0]), dtype=int)
        # Map the valence and arousal to the 5 classes
        for i in range(labels.shape[0]):
            valence, arousal = labels[i]
            if valence <= valence_thresholds[0] and arousal <= arousal_thresholds[0]:
                classes[i] = 0  # Sad
            elif valence <= valence_thresholds[0] and arousal > arousal_thresholds[1]:
                classes[i] = 1  # Angry
            elif valence > valence_thresholds[1] and arousal <= arousal_thresholds[0]:
                classes[i] = 2  # Pleasant
            elif valence > valence_thresholds[1] and arousal > arousal_thresholds[1]:
                classes[i] = 3  # Happy
            else:
                classes[i] = 4  # Neutral

        labels = classes


    main_dir = 'eeg_images'

    all_labels = []  # Initialize an empty list to store label dictionaries
    for trial_index in range(data.shape[0]):
        trial_data = data[trial_index]  # Shape (n_channels, n_samples)

        for channel_index in range(trial_data.shape[0]):
            channel_name = channels[channel_index]
            chunk_dir = os.path.join(main_dir, f'trial_{trial_index + 1}')
            if num_classes == 1:
                generate_labels(chunk_dir, channel_name, all_labels, label=labels[trial_index])
            else:
                generate_labels(chunk_dir, channel_name,  all_labels, label=labels[trial_index][0], label2=labels[trial_index][1])

    # Convert the list of label dictionaries to a DataFrame
    labels_df = pd.DataFrame(all_labels)

    # Save the labels to CSV
    labels_df.to_csv('labels.csv', index=False)
    print('CSV file created successfully.')
