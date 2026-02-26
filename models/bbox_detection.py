import torch
import torch.nn as nn
import torch.nn.functional as F

# extracts features with 3 convolution blocks, shrinks the image 3 times with pooling,
# flattens the result, then uses dense layers to predict 4 bounding box values.
# nn.Module is PyTorch's base class for all neural network models
class BBoxDetectionModel(nn.Module):
    def __init__(self):
        super(BBoxDetectionModel, self).__init__()
        
        # creating convolution layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        
        # a 2x2 window slides over the input feature map
        # reduces the spatial size of the image by half (75% reduction in total data)
        # (B, C, H, W) -> (B, C, H/2, W/2)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 32 * 32, 256)
        # maps 256 features from fc1 to 4 outputs: [x_min, y_min, width, height]
        self.fc2 = nn.Linear(256, 4)
        
    def forward(self, x):
        # extracts features, then adds non-linearity, then is shrunk spatial size by 2
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # flattening
        # ex: (B, 64, 90, 160) -> (B, 921600)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        # shape: (B, 4)
        x = self.fc2(x)
        
        # constrain outputs to valid ranges
        min_x = torch.sigmoid(x[:, 0])
        min_y = torch.sigmoid(x[:, 1])
        
        # ensures w/h is positive and less than 1 if normalized
        w = torch.sigmoid(x[:, 2])
        h = torch.sigmoid(x[:, 3])
        
        x = torch.stack([min_x, min_y, w, h], dim=1)
        return x
    
# loss function: criterion = nn.SmoothL1Loss()
# lowk what is SmoothL1 loss compared to MSE loss..
    
# for images, boxes in dataloader:

#     optimizer.zero_grad()

#     preds = model(images)

#     loss = criterion(preds, boxes)

#     loss.backward()
#     optimizer.step()

# train on 50 images first to check for overfitting, then scale up