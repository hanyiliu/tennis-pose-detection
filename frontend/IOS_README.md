# iOS Frontend Deployment of Tennis Pose Detection Model

## Overview
This deployment will be done to an iOS device using SwiftUI. The application will be primarily a Live Camera View, where inflowing frames are passed to the TPD model in real-time, with overlay of bounding boxes, keypoint frames, and final class prediction + confidence all toggleable in the app. 

Secondly, for non-realtime inference, we also allow the application to access the user's camera roll to load videos or frames for inference.

The application as a whole will look very similar to the iOS Camera app, minus the ability to take pictures and record video:
- When in the home Live Camera View, frames are processed in real-time and overlayed with model predictions that can be toggled on and off.
- When user selects the camera roll button, they are presented with the default photo/video selection screen. After selecting, they are taken to a view just like the Photo Preview View for the default iOS Photos app, where all prediction overlays are shown over this, and toggleable in the same design as the home Live Camera View.

## Frontend Implementation
The frontend application will consist of the following views, in order of priority and user activity:
1. Live Camera View: Live incoming frame feed of camera, overlaid with real-time model predictions.
    - Bottom left: Camera roll selection button, takes user to Photo Selection View.
    - Bottom center / right: Model prediction toggles for visibility of bounding box, keypoints, final class prediction, final model confidence.
2. Photo Selection View: Access to user's camera roll, allow user to import any photo or video to be used. Presented as full sheet.
    - Upon photo / video selection: User is taken to Photo Preview View.
    - Top left: Exit button, takes user back to Live Camera View.
3. Photo Preview View: Shows video / photo with all toggleable overlays. If video, show video progress bar on bottom that is scrollable.
    - Bottom left: Share button, allows export of current overlaid video to user's camera roll.
     - Bottom center / right: Model prediction toggles for visibility. Same as toggles in Live Camera View.
     - Top left: Exit button, takes user to Live Camera View.

  ## Model Export
  Export the model pipeline as a `.mlpackage`. Run model on iOS through CoreML. 