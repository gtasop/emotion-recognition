import pickle
import numpy as np

def load_data_deap(nomarlize_labels=True):
    # List to store data and labels from all subjects
    all_data = []
    all_labels = []

    for i in range(31):
        if i >= 9:
            participant = i+1
        else:
            participant = f"0{i+1}"
        with open(f"deap/data_preprocessed_python/s{participant}.dat", 'rb') as f:
            data = pickle.load(f, encoding="latin1")
            all_data.append(data['data'])
            all_labels.append(data['labels'])
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)

    all_data = all_data[:, :, 0:32, 3*128:]

    if nomarlize_labels:
        all_labels = all_labels / 9.0

    # Convert to float32 type
    all_data = all_data.astype(np.float32)
    all_labels = all_labels.astype(np.float32)

    return all_data, all_labels
