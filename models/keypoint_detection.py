"""
Keypoint Detection Model
"""

import torch
from torch import nn


class KeypointDetectionModel(nn.Module):
    """Given cropped image of player, output player keypoints.

    A Encoder-Decoder CNN model implemented in PyTorch that handles player keypoint detection.

    Model Input:
    - Only part of the image containing just the player as a (bbox height x bbox width x 3)
        matrix, 3 = # of RGB channels.

    Model Output:
    - Heatmap of each keypoint (matrix of 18 channels with each channel having
        input image width and height).

    Usage:
        # Inference
        model = KeypointDetectionModel()
        output = model(input) # Calls forward() automatically

        # Training
        model = KeypointDetectionModel()
        criterion = ...
        optimizer = ...

        model.train()
        for epoch in range(num_epochs):
            ...
    """

    def __init__(self, num_keypoints=18):
        super().__init__()

        # Encoder Stage
        self.encoder_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.encoder_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.encoder_conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.encoder_conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck_conv1 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)

        # Decoder Stage
        self.decoder_convtran1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder_conv1 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)

        self.decoder_convtran2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)

        self.decoder_convtran3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)

        self.decoder_convtran4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)

        # Extraction Stage
        self.extraction_conv1 = nn.Conv2d(64, num_keypoints, kernel_size=1, stride=1)

    def forward(self, x):
        """Forward pipeline for Keypoint Detection model.
        Args:
            x (torch.Tensor): Input 3-channel image of player.

        Returns:
            output (torch.Tensor): Heatmap of `num_keypoints`-channel, where 
            each channel shows model confidence on its keypoint.
        """
        e1 = torch.relu(self.encoder_conv1(x))
        e2 = torch.relu(self.encoder_conv2(e1))
        e3 = torch.relu(self.encoder_conv3(e2))
        e4 = torch.relu(self.encoder_conv4(e3))

        b = torch.relu(self.bottleneck_conv1(e4))

        d1 = self.decoder_convtran1(b)
        d1 = torch.cat([d1, e4], dim=1)
        d1 = torch.relu(self.decoder_conv1(d1))

        d2 = self.decoder_convtran2(d1)
        d2 = torch.cat([d2, e3], dim=1)
        d2 = torch.relu(self.decoder_conv2(d2))

        d3 = self.decoder_convtran3(d2)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = torch.relu(self.decoder_conv3(d3))

        d4 = self.decoder_convtran4(d3)
        d4 = torch.cat([d4, e1], dim=1)
        d4 = torch.relu(self.decoder_conv4(d4))

        output = self.extraction_conv1(d4)

        return output
