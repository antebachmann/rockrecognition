# Rock Recognition 
This program trains an Ultralytics YOLO model to recognize rocks from multiple sources of data.

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
    ├───runs
    │   └───detect  #output directory of trained models
    │       ├───rock_detection
    │       │   └───weights
    ├───src			# source files for backend code
    │   └───__pycache__
    ├───swissImage_50cm_patches  # Images used to train models
    ├───swissSURFACE3D_hillshade_patches
    ├───swissSURFACE3D_patches
    └───swiss_combined  # combination of the three sets of images
```