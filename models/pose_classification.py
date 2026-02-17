# models/pose_classification.py

import torch
import torch.nn as nn


class PoseClassificationModel(nn.Module):
    """
    Stage 3 Pose Classification

    Input:
        keypoints: Tensor of shape (B, 18, 3) or (18, 3)
                  each row: [x, y, visibility]
                  x,y should ideally be normalized to [0,1]

    Output:
        probs: Tensor of shape (B, 4) (softmax confidence)
        (Optional) logits: Tensor of shape (B, 4) if return_logits=True
    """

    def __init__( 
        # hyperparameters for pose classification model
        self,
        num_keypoints: int = 18,
        num_classes: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.25,
        visibility_threshold: float = 0.0,
    ):
        """
        Args:
            num_keypoints: number of keypoints (default 18)
            num_classes: number of pose classes (default 4)
            hidden_dim: hidden layer width
            dropout: dropout probability
            visibility_threshold: if > 0, keypoints with visibility below this are suppressed
                                  (helps ignore unreliable joints)
        """
        super().__init__()
        # Store hyperparameters for potential use in forward method (e.g. visibility thresholding)
        self.num_keypoints = num_keypoints
        self.num_classes = num_classes
        self.visibility_threshold = visibility_threshold

        in_dim = num_keypoints * 3  # 54, first linear layer needs a fixed input dimension, which is num_keypoints * 3 (x,y,visibility for each keypoint)

        self.fc1 = nn.Linear(in_dim, hidden_dim) # fully connected layer that takes in flattened keypoints and outputs hidden_dim features
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # another fully connected layer that takes in hidden_dim features and outputs hidden_dim features (allows for more complex representations)
        self.out = nn.Linear(hidden_dim, num_classes) # final output layer that takes in hidden_dim features and outputs num_classes logits for classification
        self.drop = nn.Dropout(dropout)

    def _apply_visibility_mask(self, keypoints: torch.Tensor) -> torch.Tensor:
        """
        Optionally suppress low-visibility keypoints by zeroing x,y,vis when vis < threshold.
        
        EX) if wrist/ankles are mossing, don't use bad coordinates to try to make up for the fact and create noise.
        """
        if self.visibility_threshold <= 0: # if threshold is 0 or negative, we don't apply any masking and just return the original keypoints.
            return keypoints

        xy = keypoints[..., 0:2]          # (B,18,2)
        vis = keypoints[..., 2:3]         # (B,18,1)
        mask = (vis >= self.visibility_threshold).float() # (B,18,1), 1 where visibility is above threshold, 0 where it's below. This mask will be used to zero out low-visibility keypoints.

        xy = xy * mask # zero out x and y where visibility is below threshold
        vis = vis * mask # zero out visibility where it's below threshold (not strictly necessary since vis is already below threshold, but keeps the output consistent)
        return torch.cat([xy, vis], dim=-1) # (B,18,3), with low-visibility keypoints zeroed out

    def forward(self, keypoints: torch.Tensor, return_logits: bool = False) -> torch.Tensor: # strict shape check.
        # Accept single sample (18,3)
        if keypoints.dim() == 2: # if input is (18,3), add batch dimension to make it (1,18,3) so it can be processed by the model. This allows the model to handle both single samples and batches of samples.
            keypoints = keypoints.unsqueeze(0)  # (1,18,3)

        if keypoints.dim() != 3 or keypoints.size(1) != self.num_keypoints or keypoints.size(2) != 3: # ensure input has shape (B,18,3) where B is batch size, 18 is number of keypoints, and 3 is (x,y,visibility) for each keypoint. If not, raise an error.
            raise ValueError(
                f"Expected keypoints shape (B,{self.num_keypoints},3) or ({self.num_keypoints},3); "
                f"got {tuple(keypoints.shape)}"
            )

        keypoints = self._apply_visibility_mask(keypoints) # zero out low-visibility keypoints if visibility_threshold > 0, otherwise returns original keypoints unchanged.

        x = keypoints.reshape(keypoints.size(0), -1)  # (B,54), flatten keypoints into shape (B, num_keypoints*3) to be input into fully connected layers. Each sample in the batch is now represented as a 54-dimensional vector containing the x,y,visibility for each of the 18 keypoints.

        x = torch.relu(self.fc1(x))        # (B,hidden_dim), pass through first fully connected layer and apply ReLU activation to introduce non-linearity. The model will learn more complex representations of the keypoint data.
        x = self.drop(x)               # apply dropout for regularization, randomly zeroing out some of the features to prevent overfitting and encourage the model to learn more robust features that generalize better to unseen data.
        x = torch.relu(self.fc2(x))        # (B,hidden_dim), pass through second fully connected layer and apply ReLU activation again for more complex representations.
        x = self.drop(x)               # apply dropout again for regularization before the final output layer.      
        logits = self.out(x)  # (B,4)

        if return_logits: # if the caller wants the raw logits (e.g. for use with a loss function like CrossEntropyLoss that expects logits), return them directly without applying softmax.
            return logits 

        probs = torch.softmax(logits, dim=-1) # (B,4), apply softmax to convert logits into probabilities for each class. The output will be a tensor of shape (B, num_classes) where each row sums to 1 and represents the model's confidence in each class for that sample.
        return probs
