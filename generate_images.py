import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
from load_data import load_data_deap
from preprocessing import binarize_array
from generate_labels import generate_labels

# Set the matplotlib backend to 'Agg' to avoid GUI issues
plt.switch_backend('Agg')


# Function to save the PSD plot for each channel in each time window
def save_channel_psd_plot(img_dir, channel_data, sampling_rate, channel_name):
    psds, freqs = mne.time_frequency.psd_array_multitaper(channel_data, sfreq=sampling_rate, fmin=0.1, fmax=40)
    img_name = f'channel_{channel_name}_psd.png'
    img_path = os.path.join(img_dir, img_name)

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, 10 * np.log10(psds), color='black')  # Plot in black for grayscale
    plt.axis('off')  # Remove axes
    plt.savefig(img_path, dpi=150, bbox_inches='tight', format='png')
    plt.close()


def generate_images(original_classes=True, valence=True, arousal=False):
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


    # Sampling rate
    sampling_rate = 128  # Adjust according to your data

    # Ensure the main directory for saving images exists
    main_dir = 'eeg_images'
    os.makedirs(main_dir, exist_ok=True)


    all_labels = []  # Initialize an empty list to store label dictionaries
    for trial_index in range(data.shape[0]):
        trial_data = data[trial_index]  # Shape (n_channels, n_samples)
        print(f'generating images for trial {trial_index + 1}/{data.shape[0]}')

        for channel_index in range(trial_data.shape[0]):
            channel_data = trial_data[channel_index]
            channel_name = channels[channel_index]
            chunk_dir = os.path.join(main_dir, f'trial_{trial_index + 1}')
            os.makedirs(chunk_dir, exist_ok=True)
            save_channel_psd_plot(chunk_dir, channel_data, sampling_rate, channels[channel_index])
            if num_classes == 1:
                generate_labels(chunk_dir, channel_name, all_labels, label=labels[trial_index])
            else:
                generate_labels(img_dir=chunk_dir, channel_name=channel_name, label=labels[trial_index][0],
                                label2=labels[trial_index][1])

    # Convert the list of label dictionaries to a DataFrame
    labels_df = pd.DataFrame(all_labels)

    # Save the labels to CSV
    labels_df.to_csv('labels.csv', index=False)
    print('CSV file created successfully.')
