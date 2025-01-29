import numpy as np
import scipy.io


def replace_nan_with_mean(data):
    # Calculate the mean of each column, ignoring NaNs
    col_mean = np.nanmean(data, axis=0)
    # Find indices where NaNs are present
    indices = np.where(np.isnan(data))
    # Replace NaNs with the mean of the corresponding column
    data[indices] = np.take(col_mean, indices[1])
    return data

def load_data_amigos():
    all_data = []
    all_labels = []
    for i in range(1, 41):
        if i < 10:
            file = f"Data_Preprocessed_P0{i}"
        else:
            file = f"Data_Preprocessed_P{i}"
        data = scipy.io.loadmat(f'amigos/preprocessed/{file}/{file}.mat')
        all_data.append(data['joined_data'][0])
        all_labels.append(data['labels_selfassessment'][0])

    # Convert to numpy array and keep the first 16 videos which are short form
    all_data = np.array(all_data)[:, :16]
    all_labels = np.array(all_labels)[:, :16]


    # keep only the 7680 timesteps to match the DEAP data
    target_shape = (40, 16, 7680, 17)
    padded_array = np.zeros(target_shape, dtype=all_data.dtype)
    for i in range(40):
        for j in range(16):
            x_len = min(all_data[i, j].shape[0], 7680)
            padded_array[i, j, :x_len, :] = all_data[i, j][:x_len, :]
    all_data = padded_array
        # Invert the last two dimensions to get shape (40, 16, 17, 8064)
    all_data = np.transpose(all_data, (0, 1, 3, 2))
    # Keep only the 14 first channels
    all_data = all_data[:, :, :14, :]
    # Extract the inner arrays
    target_array = np.stack([[all_labels[i, j].reshape(12) for j in range(16)] for i in range(40)])
    all_labels = target_array
    # Select the first 4 emotions
    all_labels = all_labels[:, :, :4]
    # Invert the first two emotions to match the DEAP dataset
    all_labels[:, :, [0, 1]] = all_labels[:, :, [1, 0]]

    # Normalize labels
    all_labels = all_labels / 9.0

    # Convert to float32 type
    all_data = all_data.astype(np.float32)
    all_labels = all_labels.astype(np.float32)

    all_data = replace_nan_with_mean(all_data.copy())
    all_labels = replace_nan_with_mean(all_labels.copy())


    contains_nan_data = np.isnan(all_data).any()
    contains_nan_labels = np.isnan(all_labels).any()
    print("AMIGOS Contains NaN values:", contains_nan_data+contains_nan_labels)

    return all_data, all_labels
