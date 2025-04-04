import torch.nn as nn
import torchvision.models as models

# Define emotion classifier model
class EmotionModel(nn.Module):
    def __init__(self, num_classes):
        super(EmotionModel, self).__init__()
        self.model = models.resnet18(pretrained=True)  # Load ResNet18
        self.model.fc = nn.Linear(512, num_classes)  # Replace final layer

    def forward(self, x):
        return self.model(x)  # Forward pass
