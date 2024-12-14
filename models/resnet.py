import torch.nn as nn
from torchvision.models import resnet18 as torch_resnet18
from torchvision.models import resnet50 as torch_resnet50


class resnet18(nn.Module):
    def __init__(self, num_classes, pretrained):
        super().__init__()

        self.backbone = torch_resnet18(pretrained=pretrained)
        
        # remove the last fc layer
        self.output_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential()

        # set new fc layer
        self.fc = nn.Linear(self.output_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features)
        return logits, features
    
    def get_embedding_dim(self):
        return self.output_dim  # 512 for resnet18


class resnet50(nn.Module):
    def __init__(self, num_classes, pretrained):
        super().__init__()

        self.backbone = torch_resnet50(pretrained=pretrained)
        
        # remove the last fc layer
        self.output_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential()

        # set new fc layer
        self.fc = nn.Linear(self.output_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features)
        return logits, features
    
    def get_embedding_dim(self):
        return self.output_dim 