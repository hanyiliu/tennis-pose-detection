# Web Frontend Deployment of Tennis Pose Detection Model

## Overview
This deployment will be done on web using Angular. The application will allow user to upload any image, of which we then pass as a REST API request to a locally hosted Python server through FastEndpoint that will process this and return model results, which is also overlayed on top of the input image. 

## Frontend Implementation
The application will have one page. Upon landing, user has option to upload picture file. Once uploaded, we make a FastEndpoint request to locally hosted Python server, which returns model values and overlayed image with predictions.

After model processed the image, we also show the numerical predictions of the model at each stage below the final predicted image.

The raw input image and the model overlayed image will be horizontally next to each other.

The user still has the option to upload a different image after uploading one.

## Model Access
The model will be accessed through a FastEndpoint request. This will be local. The only request to be made will be a `GET` request, where:  
Input: Upload a raw image  
Output: Return the overlayed image with model predictions, numerical stage 1 model output, numerical stage 2 model output, and numerical stage 3 model output.  

Numerical stage 1 output: Bbox values [X, Y, width, height].  
Numerical stage 2 output: Heatmaps of each keypoint, unified onto one graph.  
Numerical stage 3 output: Prediction confidence of each class, presented as a graph where X = class and Y = model confidence.  
