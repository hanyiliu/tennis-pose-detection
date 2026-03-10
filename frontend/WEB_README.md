# Web Frontend Deployment of Tennis Pose Detection Model

## Overview
This deployment will be done on web using Angular. The application will allow user to upload any image, of which we then pass as a REST API request to a locally hosted Python server through FastEndpoint that will process this and return model results, which is also overlayed on top of the input image. 

## Frontend Implementation
The application will have one page. Upon landing, user has option to upload picture file. Once uploaded, we make a FastEndpoint request to locally hosted Python server, which returns model values and overlayed image with predictions.

### Frontend Main Page
The main page will be built of the following:
- Top navbar, which has on left, Tennis Pose Detection title, then on right the section titles Prediction, Stage One, Stage Two, Stage Three. The section titles are hidden when no image is uploaded. (`navbar` component)
- When no uploaded image:
    - Upload image button centered on app. Only navbar, upload image button, and footer are visible when no images are uploaded.  (`import-image-button` component)
- When uploaded image, the below are all shown, which all fall into the `model-predictions` component:
    - Make another prediction button, which removes uploaded image and returns page to state of no uploaded image. (`back-button` component)
    - Final Prediction section, where section header is "Model Prediction", and we place original image to left, and overlayed model prediction image to right.  (`model-final-prediction` component)
    - Stage 1 Prediction section, where section header is "Stage 1, Bounding Box Detection", and we place bounding box numerical values on left, which are vertically stacked, and cropped bbox image on right. (`model-prediction-item` component)
    - Stage 2 Prediction section, where section header is "Stage 2, Keypoint Detection", and we place keypoint numerical values on left, which are vertically stacked, and both keypoint heatmap and keypoint overlayed image, vertically stacked, on right. (`model-prediction-item` component)
    - Stage 3 Prediction section, where section header is "Stage 3 - Pose Classification", and we place pose classification numerical values on left, which are vertically stacked, and bar graph of class confidences on right. (`model-prediction-item` component)
- Footer, which has centered and in order, the names of the contributors, link to Github, copyright.  (`footer` component)

Supporting components:
- Each section (`model-final-prediction`, `model-prediction-item`) all uses a section header, which is the `section-header` component. This component is a two-item title, with main title on left and sub title on right. To the right of the sub title, there also is a button that collapses and expands the current section and all items in it. The expand and collapse should be animated and smooth. This expand/collapse button can be hidden, in which case the section cannot be collapsed and will always be expanded. Only the `model-final-prediction` component will have the expand/collapse button be hidden; all other sections are collapsable. For example, stage 1's "Stage 1, Bounding Box Detection" has "Stage 1" as main title and "Bounding Box Detection" as sub title. The two titles are the same size, but the main is aligned left while sub title is aligned right.
- Each image is displayed using the `image-card` component. This component simply displays the image with a rounded black border.
- All prediction section (`model-final-prediction`, `model-prediction-item`) + `back-button` go into the `model-predictions` component. This component is only shown when there is an image uploaded, otherwise it is hidden.
- For each prediction stage of `model-prediction-item`, the Prediction model defines the exact numerical values and images that should be shown in each stage.

Website theme:
- Primary color: RGB(11, 10, 152)
- Secondary color: RGB(175, 176, 222)
- Accent color: RGB(215, 184, 253)
- Background color: RGB(246, 248, 255)
- Primary font: IBM Plex Mono

## Model Access
The model will be accessed through a FastEndpoint request. This will be local. The only request to be made will be a `POST` request to `/predict`, where:  
Input body: Raw image  
Output: Return the overlayed image with model prediction values and images.

Model predictions:  
- Stage One
    - bbox_x: number
    - bbox_y: number
    - bbox_width: number
    - bbox_height: number
    - cropped_bbox_image: image (Cropped image according to bbox boundaries)

- Stage Two
    - keypoint_positions: number[][] ([[Keypoint 1 x, Keypoint 1 y, Keypoint 1 visibility], [Keypoint 2 x, Keypoint 2 y, Keypoint 2 visibility], ..., [Keypoint 18 x, Keypoint 18 y, Keypoint 18 visibility]])
    - keypoint_aggregated_heatmap_url: string (Image of all keypoint heatmaps displayed against black background.)
    - keypoint_cropped_image: image (Image of cropped image with keypoints overlayed)

- Stage Three
    - backhand_conf: number
    - forehand_conf: numbe
    - ready_position_conf: number
    - serve_conf: number
    - class_prediction_bar_graph: image (Bar graph of class prediction confidences)

- Final Prediction
    - overlayed_image: image (Overlayed image with bounding box, keypoints, and final class prediction)

These predictions map directly to `tpd-demo`'s Prediction model.
