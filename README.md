﻿# Rock Recognition 
This program trains an Ultralytics YOLO model to recognize rocks from multiple sources of data.
First use earlyfusiontest.py to handle data then use main.py to train a model.
To do inference, first unzip the models then run the notebook.
You need to have the three sets of data to make it work.
File structure:
```
─rockrecognition
    ├───datasets
    │   ├───annotations
    │   ├───train
    │   │   ├───images
    │   │   └───labels
    │   └───val
    │       ├───images
    │       └───labels
    ├───models      #Where models are stored for inference
    │
    ├───runs
    │   └───detect  #output directory of trained models
    │       ├───rock_detection
    │       │   └───weights
    ├───src			# source files for backend code
    │   
    ├───swissImage_50cm_patches  # Images used to train models
    ├───swissSURFACE3D_hillshade_patches
    ├───swissSURFACE3D_patches
    └───* images  # processed images after earlyfusion is run
```
