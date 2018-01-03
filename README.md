
## Vehicle Detection
### This project is to detect the vehicle on the road using the dash camera on the car

The code is available in [Vehicle_Detection.ipynb](Vehicle_Detection.ipynb).

---
Algorithm Used: You Only Look Once (YOLO) v1
---------
## Introduction to object detection

Detecting vehicles in a video stream is an object detection problem. An object detection problem can be approached as either a classification problem or a regression problem. As a classification problem, the image are divided into small patches, each of which will be run through a classifier to determine whether there are objects in the patch. Then the bounding boxes will be assigned to locate around patches that are classified with high probability of present of an object. In the regression approach, the whole image will be run through a convolutional neural network to directly generate one or more bounding boxes for objects in the images.

| classification                                                                                                                               | regression                                               |
|----------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|
| Classification on portions of the image to determine objects, generate bounding boxes for regions that have positive classification results. | Regression on the whole image to generate bounding boxes |
| 1. sliding window + HOG 2. sliding window + CNN 3. region proposals + CNN                                                                    | generate bounding box coordinates directly from CNN      |
| RCNN, Fast-RCNN, Faster-RCNN                                                                                                                 | SSD, YOLO                                                |

In this project, we will use tiny-YOLO v1, since it's easy to implement and are reasonably fast.


## The tiny-YOLO v1

### Architecture of the convolutional neural network

The tiny YOLO v1 is consist of 9 convolution layers and 3 full connected layers. Each convolution layer consists of convolution, leaky relu and max pooling operations. The first 9 convolution layers can be understood as the feature extractor, whereas the last three full connected layers can be understood as the "regression head" that predicts the bounding boxes.

![model](./output_images/mode_yolo_plot.jpg)

There are a total of 45,089,374 parameters in the model and the detail of the architecture is in list in this table


    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    convolution2d_1 (Convolution2D)  (None, 16, 448, 448)  448         convolution2d_input_1[0][0]      
    ____________________________________________________________________________________________________
    leakyrelu_1 (LeakyReLU)          (None, 16, 448, 448)  0           convolution2d_1[0][0]            
    ____________________________________________________________________________________________________
    maxpooling2d_1 (MaxPooling2D)    (None, 16, 224, 224)  0           leakyrelu_1[0][0]                
    ____________________________________________________________________________________________________
    convolution2d_2 (Convolution2D)  (None, 32, 224, 224)  4640        maxpooling2d_1[0][0]             
    ____________________________________________________________________________________________________
    leakyrelu_2 (LeakyReLU)          (None, 32, 224, 224)  0           convolution2d_2[0][0]            
    ____________________________________________________________________________________________________
    maxpooling2d_2 (MaxPooling2D)    (None, 32, 112, 112)  0           leakyrelu_2[0][0]                
    ____________________________________________________________________________________________________
    convolution2d_3 (Convolution2D)  (None, 64, 112, 112)  18496       maxpooling2d_2[0][0]             
    ____________________________________________________________________________________________________
    leakyrelu_3 (LeakyReLU)          (None, 64, 112, 112)  0           convolution2d_3[0][0]            
    ____________________________________________________________________________________________________
    maxpooling2d_3 (MaxPooling2D)    (None, 64, 56, 56)    0           leakyrelu_3[0][0]                
    ____________________________________________________________________________________________________
    convolution2d_4 (Convolution2D)  (None, 128, 56, 56)   73856       maxpooling2d_3[0][0]             
    ____________________________________________________________________________________________________
    leakyrelu_4 (LeakyReLU)          (None, 128, 56, 56)   0           convolution2d_4[0][0]            
    ____________________________________________________________________________________________________
    maxpooling2d_4 (MaxPooling2D)    (None, 128, 28, 28)   0           leakyrelu_4[0][0]                
    ____________________________________________________________________________________________________
    convolution2d_5 (Convolution2D)  (None, 256, 28, 28)   295168      maxpooling2d_4[0][0]             
    ____________________________________________________________________________________________________
    leakyrelu_5 (LeakyReLU)          (None, 256, 28, 28)   0           convolution2d_5[0][0]            
    ____________________________________________________________________________________________________
    maxpooling2d_5 (MaxPooling2D)    (None, 256, 14, 14)   0           leakyrelu_5[0][0]                
    ____________________________________________________________________________________________________
    convolution2d_6 (Convolution2D)  (None, 512, 14, 14)   1180160     maxpooling2d_5[0][0]             
    ____________________________________________________________________________________________________
    leakyrelu_6 (LeakyReLU)          (None, 512, 14, 14)   0           convolution2d_6[0][0]            
    ____________________________________________________________________________________________________
    maxpooling2d_6 (MaxPooling2D)    (None, 512, 7, 7)     0           leakyrelu_6[0][0]                
    ____________________________________________________________________________________________________
    convolution2d_7 (Convolution2D)  (None, 1024, 7, 7)    4719616     maxpooling2d_6[0][0]             
    ____________________________________________________________________________________________________
    leakyrelu_7 (LeakyReLU)          (None, 1024, 7, 7)    0           convolution2d_7[0][0]            
    ____________________________________________________________________________________________________
    convolution2d_8 (Convolution2D)  (None, 1024, 7, 7)    9438208     leakyrelu_7[0][0]                
    ____________________________________________________________________________________________________
    leakyrelu_8 (LeakyReLU)          (None, 1024, 7, 7)    0           convolution2d_8[0][0]            
    ____________________________________________________________________________________________________
    convolution2d_9 (Convolution2D)  (None, 1024, 7, 7)    9438208     leakyrelu_8[0][0]                
    ____________________________________________________________________________________________________
    leakyrelu_9 (LeakyReLU)          (None, 1024, 7, 7)    0           convolution2d_9[0][0]            
    ____________________________________________________________________________________________________
    flatten_1 (Flatten)              (None, 50176)         0           leakyrelu_9[0][0]                
    ____________________________________________________________________________________________________
    dense_1 (Dense)                  (None, 256)           12845312    flatten_1[0][0]                  
    ____________________________________________________________________________________________________
    dense_2 (Dense)                  (None, 4096)          1052672     dense_1[0][0]                    
    ____________________________________________________________________________________________________
    leakyrelu_10 (LeakyReLU)         (None, 4096)          0           dense_2[0][0]                    
    ____________________________________________________________________________________________________
    dense_3 (Dense)                  (None, 1470)          6022590     leakyrelu_10[0][0]               
    ====================================================================================================
    Total params: 45,089,374
    Trainable params: 45,089,374
    Non-trainable params: 0
    ____________________________________________________________________________________________________


In this project, we will use Keras to construct the YOLO model.

Implementation
---

## Pre-processing

### Area of interest, cropping and resizing
Input to the model is a batch of 448x448 images. So we first determine the area of interest for each image. We only consider this portion of the image for prediction, since cars won't be present all over the image, just on the roads in the lower portion of the image. Then this cropped image is resized to a 448x448 image.

### Normalization

Each image pixel is normalized to have values between -1 and 1. We use simple min-max normalization to achieve this.

## Training

I have used pre-trained weights for this project. Training is done in 2 parts

### Part 1: Training for classification
This model was trained on ImageNet 1000-class classification dataset. For this we take the first 6 convolutional layers followed by a followed by a fully connected layer.

### Part 2: Training for detection
The model is then converted for detection. This is done by adding 3 convolutional layers and 3 fully connected layers. The modified model is then trained on PASCAL VOC detection dataset.

The pre-trained weights for this model (180 MB) are available [here](https://www.dropbox.com/s/92ab1504jctuva7/yolo-tiny.weights?dl=0).

![png](output_images/yolo.png)

## Post Processing

The model was trained on PASCAL VOC dataset. We use S = 7, B = 2. PASCAL VOC has 20 labelled classes so C = 20. So our final prediction, for each input image, is:

<img src="./output_images/net_output.png" width="600"/>

    output tensor length = S x S x (B x 5 + C)
    output tensor length = 7 x 7 x (2x5 + 20)
    output tensor length = 1470.

The structure of the 1470 length tensor is as follows:

1. First 980 values correspons to probabilities for each of the 20 classes for each grid cell. These probabilities are conditioned on objects being present in each grid cell.
2. The next 98 values are confidence scores for 2 bounding boxes predicted by each grid cells.
3. The next 392 values are co-ordinates (x, y, w, h) for 2 bounding boxes per grid cell.

As you can see in the above image, each input image is divided into an S x S grid and for each grid cell, our model predicts B bounding boxes and C confidence scores. There is a fair amount of post-processing involved to arrive at the final bounding boxes based on the model's predictions.

### Class score threshold
We reject output from grid cells below a certain threshold (0.2) of class scores (equation 2), computed at test time.

### Reject overlapping (duplicate) bounding boxes
If multiple bounding boxes, for each class overlap and have an IOU of more than 0.4 (intersecting area is 40% of union area of boxes), then we keep the box with the highest class score and reject the other box(es).

### Drawing the bounding boxes
The predictions (x, y) for each bounding box are relative to the bounds of the grid cell and (w, h) are relative to the whole image. To compute the final bounding box coodinates we have to multiply `w` & `h` with the width & height of the portion of the image used as input for the network.

Testing
---

The pipeline is applied to individual images. Here is the result.

![png](output_images/test_output.png)

The Video
---
The pipeline is applied to a video. Click on the image to watch the video or [click here](https://youtu.be/ahe2pcyWUFw). You will be redirected to YouTube.

[![Project Video](https://i.ytimg.com/vi/ahe2pcyWUFw/hqdefault.jpg)](https://youtu.be/ahe2pcyWUFw)



## Discussion

The YOLO is known to be fast. In the original paper, the tiny-YOLO is reported to work at nearly 200 FPS on a powerful desktop GPU. In this project, the video is processed on a Nvidia 1070 and the rate is about 21FS without batch processing.

## Reference

1. J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, You Only Look Once: Unified, Real-Time Object Detection, arXiv:1506.02640 (2015).
2. J. Redmon and A. Farhadi, YOLO9000: Better, Faster, Stronger, arXiv:1612.08242 (2016).
3. darkflow, https://github.com/thtrieu/darkflow
4. Darknet.keras, https://github.com/sunshineatnoon/Darknet.keras/
5. YAD2K, https://github.com/allanzelener/YAD2K
6. Yolo website, https://pjreddie.com/darknet/yolo/
