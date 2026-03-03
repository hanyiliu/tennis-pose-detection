import torch
import torch.nn as nn
import torchvision.models as models


class BBoxDetectionModel(nn.Module):
    # pretrained ResNet18 backbone + small regression head
    # outputs normalized bbox [min_x, min_y, w, h] in [0, 1]

    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()

        # ResNet18
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
        except Exception:
            backbone = models.resnet18(pretrained=pretrained)

        in_feats = backbone.fc.in_features
        backbone.fc = nn.Identity() # remove classifier
        self.backbone = backbone

        # regression head
        self.head = nn.Sequential(
            nn.Linear(in_feats, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 4),
        )

    def forward(self, x):
        feats = self.backbone(x) # (B, in_feats)
        out = self.head(feats) # (B, 4)

        # map to [0,1]
        min_x = torch.sigmoid(out[:, 0])
        min_y = torch.sigmoid(out[:, 1])

        # positive width/height in [0,1]
        w = torch.sigmoid(out[:, 2])
        h = torch.sigmoid(out[:, 3])

        return torch.stack([min_x, min_y, w, h], dim=1)