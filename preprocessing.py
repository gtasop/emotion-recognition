import numpy as np


def binarize_array(arr, threshold=0.5):
  if not isinstance(arr, np.ndarray):
    raise TypeError("Input must be a NumPy array")

  binarized_arr = (arr >= threshold).astype(int)
  return binarized_arr

def segment(data, labels, chunk_size):
  segmented_data = []
  segmented_labels = []

  for subject in range(data.shape[0]):
    subject_data = []
    subject_labels = []
    for trial in range(data.shape[1]):
      time_series = data[subject, trial]
      num_chunks = time_series.shape[-1] // chunk_size

      for chunk in range(num_chunks):
        start = chunk * chunk_size
        end = start + chunk_size
        subject_data.append(time_series[:, start:end])
        subject_labels.append(labels[subject, trial])

    segmented_data.append(subject_data)
    segmented_labels.append(subject_labels)

  segmented_data = np.array(segmented_data)
  segmented_labels = np.array(segmented_labels)

  return segmented_data, segmented_labels