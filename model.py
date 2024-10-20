import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super(EfficientNetV2, self).__init__()
        self.model = models.efficientnet_v2_s(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
