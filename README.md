# Lightweight object detection tracking model for traffic counting

This repo mainly consists of two parts, the first part is related to the light weight object detection models, which are used to detect traffic participants. And the second part is about object tracking based on the detection results of the first part.

1. The detection models are implemented using Keras, in the folder 'notebooks_train_detection_model' contains all the jupyter notebooks used for training the detection models. The scripts need to be move to the root folder at first and then can be used. 
The pretrained weights file in the 'trained_weights' folder; The model structure in the 'model' folder; The scripts that are used to generate datasets statistic charts are saved in folder 'scripts_draw_charts'; The folder 'data_index' contains the data consumed by detection model.

2. The tracker logic and related files are saved in the folder 'tracker', please follow the README in the folder to run the tracker, and the related data is also saved in the folder. The scripts related to the dynamic velocity estimation model experiments are saved in the folder 'scripts_related_to_experiments_of_dynamic_velocity_model'.

## Dependencies:
* Python 3.x
* Numpy
* TensorFlow 1.x
* Keras 2.x
* OpenCV
* Beautiful Soup 4.x

