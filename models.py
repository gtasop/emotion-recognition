import torch.nn as nn
import torchvision.models as models
from torcheeg import models as models_torcheeg



class ModifiedEEGNet(nn.Module):
    def __init__(self, num_classes, input_size):
        super(ModifiedEEGNet, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        self.eegnet = models_torcheeg.EEGNet(chunk_size=input_size, num_electrodes=32, num_classes=num_classes)

        self.expand_channels = nn.Conv2d(14, 32, kernel_size=1)

    def forward(self, x):
        if x.size(2) == 14:  # Check if the number of channels is 14 coming fromn AMIGOS dataset
            # Expand channels to 32
            x = x.permute(1, 2, 0, 3)
            x = self.expand_channels(x)
            x = x.permute(2, 0, 1, 3)

        elif x.size(2) != 32:
            raise ValueError("Unsupported number of channels")

        x = self.eegnet(x)

        return x


class CustomEfficientNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomEfficientNetV2, self).__init__()
        # Load EfficientNetV2 with default weights
        self.effnet = models.efficientnet_v2_s(weights='IMAGENET1K_V1')

        # Adjust the classifier for the number of output classes
        in_features = self.effnet.classifier[1].in_features
        self.effnet.classifier[1] = nn.Linear(in_features, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.effnet(x)
        return x
