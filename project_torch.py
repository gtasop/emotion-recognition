import torch
from torch import nn, optim
from training import training
from preprocessing import binarize_array, segment
from load_data import load_data_deap
from load_data_amigos import load_data_amigos
from dataset import split_datasets_across_subjects, split_datasets_within_subjects
from models import ModifiedEEGNet


def EEGNet(num_epochs=100, valence=True, arousal=True, deap=True, amigos=True, cross_subject=True, use_scheduler=True):
    torch.cuda.empty_cache()

    data_deap, labels_deap = load_data_deap()
    data_amg, labels_amg = load_data_amigos()

    # Preprocessing
    labels_deap = binarize_array(labels_deap)
    labels_amg = binarize_array(labels_amg)

    # Select just valence
    if valence and not(arousal):
        labels_deap = labels_deap[:, :, 0]
        labels_amg = labels_amg[:, :, 0]
        num_classes = 1
    # Select just arousal
    elif not(valence) and arousal:
        labels_deap = labels_deap[:, :, 1]
        labels_amg = labels_amg[:, :, 1]
        num_classes = 1
    # Select valence and arousal
    else:
        labels_deap = labels_deap[:, :, :2]
        labels_amg = labels_amg[:, :, :2]
        num_classes = 2

    # Segment timesteps into 1 second slices
    input_length = 128
    data_deap, labels_deap = segment(data_deap, labels_deap, input_length)
    data_amg, labels_amg = segment(data_amg, labels_amg, input_length)


    # Convert to tensors
    data_tensor_deap = torch.from_numpy(data_deap)
    labels_tensor_deap = torch.from_numpy(labels_deap).to(torch.float32)
    data_tensor_amg = torch.from_numpy(data_amg)
    labels_tensor_amg = torch.from_numpy(labels_amg).to(torch.float32)

    # Creating the sampler
    batch_size = 1024


    data_1, labels_1, data_2, labels_2 = None, None, None, None
    if deap and not(amigos):
        data_1 = data_tensor_deap
        labels_1 = labels_tensor_deap
    elif not(deap) and amigos:
        data_1 = data_tensor_amg
        labels_1 = labels_tensor_amg
    else:
        data_1 = data_tensor_deap
        labels_1 = labels_tensor_deap
        data_2 = data_tensor_amg
        labels_2 = labels_tensor_amg

    if cross_subject:
        train_loader, val_loader, test_loader = split_datasets_across_subjects(data1=data_1, labels1=labels_1, batch_size=batch_size,
                                                                               data2=data_2, labels2=labels_2)
    else:
        train_loader, val_loader, test_loader = split_datasets_within_subjects(data1=data_1, labels1=labels_1,
                                                                               data2=data_2, labels2=labels_2)

    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # Initialize the model, loss function, and optimizer
    model = ModifiedEEGNet(num_classes=num_classes, input_size=input_length).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20)


    training(num_epochs=num_epochs, model=model, train_loaders=train_loader, val_loaders=val_loader, test_loaders=test_loader, criterion=criterion,
             optimizer=optimizer, device=device, scheduler=scheduler, num_classes=num_classes)
