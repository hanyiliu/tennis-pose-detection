import torch
import torch.nn as nn
import torchvision.models as models


class TennisPoseClassifier(nn.Module):
    # pretrained ResNet18 classifier for classes forehand, backhand, ready_position, serve

    # 0.3 dropout regularization = 30% of activations coming into a layer are randomly set to 0
    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()

        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
        except Exception:
            backbone = models.resnet18(pretrained=pretrained)

        in_feats = backbone.fc.in_features

        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_feats, num_classes)
        )

        self.model = backbone

    def forward(self, x):
        return self.model(x) # raw logits of shape (B, 4)