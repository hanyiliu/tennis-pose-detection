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
        hidden_dim: int = 512,
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

        raw_dim = num_keypoints * 3  # 54 = 18 * (x,y,visibility)

        # visibility-weighted x,y features contribute num_keypoints * 2 features
        weighted_dim = num_keypoints * 2  # 36 = 18 * (x,y) after multiplying by visibility

        # centered x,y features contribute num_keypoints * 2 features
        centered_dim = num_keypoints * 2  # 36 = 18 * (x,y) after subtracting body center

        # global pose spread features = [min_x, min_y, spread_x, spread_y]
        global_dim = 4

        # final feature dimension combines raw keypoints + weighted coordinates + centered coordinates + global spread
        in_dim = raw_dim + weighted_dim + centered_dim + global_dim  # 54 + 36 + 36 + 4 = 130

        self.fc1 = nn.Linear(in_dim, hidden_dim)  # fully connected layer that takes in engineered pose features and outputs hidden_dim features
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # another fully connected layer that takes in hidden_dim features and outputs hidden_dim features (allows for more complex representations)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)  # compress representation slightly before final output layer
        self.out = nn.Linear(hidden_dim // 2, num_classes)  # final output layer that takes in hidden_dim//2 features and outputs num_classes logits for classification
        self.drop = nn.Dropout(dropout)

    def _apply_visibility_mask(self, keypoints: torch.Tensor) -> torch.Tensor:
        """
        Optionally suppress low-visibility keypoints by zeroing x,y,vis when vis < threshold.
        
        EX) if wrist/ankles are mossing, don't use bad coordinates to try to make up for the fact and create noise.
        """
        if self.visibility_threshold <= 0: # if threshold is 0 or negative, we don't apply any masking and just return the original keypoints.
            return keypoints

        xy = keypoints[..., 0:2]    # (B,18,2)
        vis = keypoints[..., 2:3]   # (B,18,1)
        mask = (vis >= self.visibility_threshold).float() # (B,18,1), 1 where visibility is above threshold, 0 where it's below. This mask will be used to zero out low-visibility keypoints.

        xy = xy * mask # zero out x and y where visibility is below threshold
        vis = vis * mask # zero out visibility where it's below threshold (not strictly necessary since vis is already below threshold, but keeps the output consistent)
        return torch.cat([xy, vis], dim=-1) # (B,18,3), with low-visibility keypoints zeroed out

    def _build_pose_features(self, keypoints: torch.Tensor) -> torch.Tensor:
        """
        Build a richer pose feature vector before feeding into the classifier.

        Uses:
            1) raw normalized keypoints [x, y, vis]
            2) visibility-weighted x,y coordinates
            3) body-centered x,y coordinates
            4) simple global pose spread features

        This helps the classifier focus more on reliable joints and relative pose geometry
        instead of only absolute position.
        """
        xy = keypoints[..., 0:2]   # (B,18,2), raw x and y coordinates
        vis = keypoints[..., 2:3]  # (B,18,1), visibility / confidence values

        # visibility-weighted coordinates reduce the impact of unreliable joints
        weighted_xy = xy * vis  # (B,18,2), low-confidence joints contribute less to the final feature vector

        # compute visibility-weighted body center so highly visible joints influence the center more
        vis_sum = vis.sum(dim=1, keepdim=True).clamp_min(1e-6)  # (B,1,1), clamp prevents division by zero if all joints are invisible
        center = (xy * vis).sum(dim=1, keepdim=True) / vis_sum  # (B,1,2), weighted center of the visible joints

        # subtract body center from each x,y so the pose becomes more relative and less sensitive to crop shifts
        centered_xy = xy - center  # (B,18,2)

        # compute rough pose extent using only visible joints
        big = torch.full_like(xy, 1e6)     # large placeholder used when masking invisible joints for min
        small = torch.full_like(xy, -1e6)  # small placeholder used when masking invisible joints for max

        masked_min_xy = torch.where(vis > 0, xy, big).amin(dim=1)   # (B,2), minimum visible x,y
        masked_max_xy = torch.where(vis > 0, xy, small).amax(dim=1) # (B,2), maximum visible x,y

        spread = masked_max_xy - masked_min_xy  # (B,2), width/height spread of the visible pose
        global_feats = torch.cat([masked_min_xy, spread], dim=-1)   # (B,4), [min_x, min_y, spread_x, spread_y]

        raw_flat = keypoints.reshape(keypoints.size(0), -1)           # (B,54), flatten original keypoints
        weighted_flat = weighted_xy.reshape(keypoints.size(0), -1)    # (B,36), flatten visibility-weighted x,y
        centered_flat = centered_xy.reshape(keypoints.size(0), -1)    # (B,36), flatten centered x,y

        features = torch.cat([raw_flat, weighted_flat, centered_flat, global_feats], dim=-1)  # final engineered feature vector
        return features

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

        x = self._build_pose_features(keypoints)  # (B,94), build richer pose features instead of only flattening raw keypoints

        x = torch.relu(self.fc1(x)) # (B,hidden_dim), pass through first fully connected layer and apply ReLU activation to introduce non-linearity. The model will learn more complex representations of the pose feature vector.
        x = self.drop(x) # apply dropout for regularization, randomly zeroing out some of the features to prevent overfitting and encourage the model to learn more robust features that generalize better to unseen data.
        x = torch.relu(self.fc2(x)) # (B,hidden_dim), pass through second fully connected layer and apply ReLU activation again for more complex representations.
        x = self.drop(x) # apply dropout again for regularization before the next layer.
        x = torch.relu(self.fc3(x)) # (B,hidden_dim//2), compress learned representation before final output layer
        x = self.drop(x) # apply dropout one more time before classification
        logits = self.out(x) # (B,4)

        if return_logits: # if the caller wants the raw logits (e.g. for use with a loss function like CrossEntropyLoss that expects logits), return them directly without applying softmax.
            return logits 

        probs = torch.softmax(logits, dim=-1) # (B,4), apply softmax to convert logits into probabilities for each class. The output will be a tensor of shape (B, num_classes) where each row sums to 1 and represents the model's confidence in each class for that sample.
        return probs