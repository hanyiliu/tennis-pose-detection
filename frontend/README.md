# Frontend Deployment of Tennis Pose Detection Model

This deployment will be done to an iOS device using SwiftUI. The application will be primarily a Live Camera View, where inflowing frames are passed to the TPD model in real-time, with overlay of bounding boxes, keypoint frames, and final class prediction all toggleable in the app. 

Secondly, for non-realtime inference, we also allow the application to access the user's camera roll to load videos or frames for inference.

The application as a whole will look very similar to the iOS Camera app, minus the ability to take pictures and record video:
- When in the home Live Camera View, frames are processed in real-time and overlayed with model predictions that can be toggled on and off.
- When user selects the camera roll button, they are presented with the default photo/video selection screen. After selecting, they are taken to a view just like the Photo Preview View for the default iOS Photos app, where all prediction overlays are shown over this, and toggleable in the same design as the home Live Camera View.