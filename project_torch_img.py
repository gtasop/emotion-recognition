import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset_img import EEGDataset
from models import CustomEfficientNetV2
from training import training_img
from generate_images import generate_images as gen_images
from generate_labels import generate_new_labels

def EfficientNet(generate_images=True, generate_labels=False, original_classes=True, valence=True, arousal=False):
    torch.cuda.empty_cache()

    if generate_images:
        gen_images(original_classes=original_classes, valence=valence, arousal=arousal)
    if generate_labels:
        generate_new_labels(original_classes=original_classes, valence=valence, arousal=arousal)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.GaussianBlur(kernel_size=3),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ToTensor()
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dataset
    dataset = EEGDataset(root_dir='eeg_images', labels_file='labels.csv', transform=transform)

    total_size = len(dataset)
    train_size = int(0.85 * total_size)
    val_size = int(0.10 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    batch_size = 32
    num_workers = 10
    num_classes = 2
    # Create DataLoader with optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)


    model = CustomEfficientNetV2(num_classes=num_classes).to(device)
    if num_classes > 2:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training and validation loops
    num_epochs = 30
    training_img(num_epochs, model, train_loader, val_loader, test_loader, optimizer, criterion, device, num_classes=num_classes,
                 scheduler=None, original_classes=original_classes)
