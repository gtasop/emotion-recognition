from torch.utils.data import Dataset, random_split, DataLoader, ConcatDataset


class EEGDataset(Dataset):
    def __init__(self, eeg_data, labels):
        self.eeg_data = eeg_data
        self.labels = labels

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        # Clone the data and label to avoid modifying the original data
        data = self.eeg_data[idx].clone()
        label = self.labels[idx].clone()

        return data, label


def split_datasets_within_subjects(data1, labels1, data2=None, labels2=None, train_pct=0.85, val_pct=0.10):
    training_dataloaders, validation_dataloaders, testing_dataloaders = [], [], []
    data1 = data1.unsqueeze(1)
    dataset_1 = EEGDataset(data1, labels1)
    if data2 is not None:
        data2 = data2.unsqueeze(1)
        dataset_2 = EEGDataset(data2, labels2)
        datasets = [dataset_1, dataset_2]
    else:
        datasets = [dataset_1]
    for dataset in datasets:
        eeg_data = dataset.eeg_data
        labels = dataset.labels

        train_dataset, val_dataset, test_dataset = [], [], []

        total_size = eeg_data.shape[2]
        train_size = int(train_pct * total_size)
        val_size = int(val_pct * total_size)
        test_size = total_size - train_size - val_size

        # Separate each subject's data
        num_subjects = eeg_data.shape[0]
        for subject in range(num_subjects):
            subject_data = EEGDataset(eeg_data[subject], labels[subject])
            subject_data.eeg_data = subject_data.eeg_data.permute(1, 0, 2, 3)
            train_dataset_subj, val_dataset_subj, test_dataset_subj = random_split(subject_data, [train_size, val_size, test_size])
            train_dataset.append(train_dataset_subj)
            val_dataset.append(val_dataset_subj)
            test_dataset.append(test_dataset_subj)

        combined_training_dataset = ConcatDataset(train_dataset)
        combined_validation_dataset = ConcatDataset(val_dataset)
        combined_testing_dataset = ConcatDataset(test_dataset)

        training_loader = DataLoader(combined_training_dataset, batch_size=train_size, shuffle=False)
        training_dataloaders.append(training_loader)
        validation_loader = DataLoader(combined_validation_dataset, batch_size=val_size, shuffle=False)
        validation_dataloaders.append(validation_loader)
        testing_loader = DataLoader(combined_testing_dataset, batch_size=test_size, shuffle=False)
        testing_dataloaders.append(testing_loader)

    return training_dataloaders, validation_dataloaders, testing_dataloaders


def split_datasets(dataset, train_prc=0.85, val_prc=0.10):
    total_size = len(dataset)
    train_size = int(train_prc * total_size)
    val_size = int(val_prc * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset

def split_datasets_across_subjects(data1, labels1, batch_size, data2=None, labels2=None):
    train_loader, val_loader, test_loader = [], [], []
    data1 = data1.flatten(0, 1).unsqueeze(1)
    labels1 = labels1.flatten(0, 1)
    dataset = EEGDataset(data1, labels1)
    train_dataset, val_dataset, test_dataset = split_datasets(dataset)

    train_loader.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
    val_loader.append(DataLoader(val_dataset, batch_size=batch_size, shuffle=False))
    test_loader.append(DataLoader(test_dataset, batch_size=batch_size, shuffle=False))

    if data2 is not None:
        data2 = data2.flatten(0, 1).unsqueeze(1)
        labels2 = labels2.flatten(0, 1)
        dataset2 = EEGDataset(data2, labels2)
        train_dataset_2, val_dataset_2, test_dataset_2 = split_datasets(dataset2)

        train_loader.append(DataLoader(train_dataset_2, batch_size=batch_size, shuffle=True))
        val_loader.append(DataLoader(val_dataset_2, batch_size=batch_size, shuffle=False))
        test_loader.append(DataLoader(test_dataset_2, batch_size=batch_size, shuffle=False))

    return train_loader, val_loader, test_loader