import os
import shutil
from PIL import Image
import numpy as np
from tifffile import tifffile as tiff
import yaml
import cv2
import re

def prepare_yolo_annotations(dataset, output_dir):
    """
    Convert all annotations in the dataset to YOLO format.
    Args:
        dataset (list): List of samples with file names and annotations.
        output_dir (str): Directory to save YOLO annotation files.
    """

    # Créer le répertoire s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    for sample_info in dataset:
        bboxes = sample_info['rocks_annotations']
        file_name = sample_info['file_name'].replace('.tif', '.txt')
        output_path = os.path.join(output_dir, file_name)
        with open(output_path, 'w') as f:
            for box in bboxes:
                x_rel, y_rel = box['relative_within_patch_location']
                width_rel = 30 / 640
                height_rel = 30 / 640
                f.write(f"0 {x_rel} {y_rel} {width_rel} {height_rel}\n")

def create_yaml_file(output_dir):
    data = {'train': f'{output_dir}/train/images',
            'val' : f'{output_dir}/train/images',
            'nc': 1,           # Number of classes
            'names': ['rock']} # Name of classes
    with open('data.yaml', 'w') as f:
        yaml.dump(data,f, sort_keys=False)

    return
    



def copy_files(file_list, src_img, src_ann, dest_img, dest_ann):
    os.makedirs(dest_img, exist_ok=True)
    os.makedirs(dest_ann, exist_ok=True)
    for file_name in file_list:
        shutil.copy(os.path.join(src_img, f"{file_name}.tif"), dest_img)
        shutil.copy(os.path.join(src_ann, f"{file_name}.txt"), dest_ann)


def normalize(img):
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    return img


def early_fusion(si_path, ss_path, hs_path, output_path, fusion_settings):
    """
    This function performs early fusion of 3 different types of images,
    RGB, Surface3D and Hillshade, each respectively merged into the RGB
    channels of a combined image. The RGB image is first turned into a gray 
    8 bit image.

    Parameters
    ----------
    si_path : Path of RGB images of dataset
    
    ss_path : Path of Surface 3D images of dataset
    
    hs_path : Path of Hillshade images of dataset
    
    output_path : Path of output directory
    

    Raises
    ------
    ValueError
        Checks if image dimensions are consistent.

    Returns
    -------
    None.

    """
    
    option_list = ['RGB', '3D', 'HS', 'R', 'G', 'B']
    
    
    fusion_settings = re.split('\s', fusion_settings)
    
    for setting in fusion_settings:
        if setting not in option_list:
            raise Exception('Setting not available or incorrectly formatted.')
            exit()
    if len(fusion_settings) == 1:
        if fusion_settings == ['RGB']:
            fusion_settings = ['R', 'G', 'B']
        else:
            fusion_settings *= 3
    
    si_img = tiff.imread(si_path)  # Image RGB (3 canaux)
    ss_img = tiff.imread(ss_path)  # Image Surface 3D (1 canal)
    hs_img = tiff.imread(hs_path)  # Image Hillshade (1 canal)
    fused_image = si_img.copy()
    
    #si_img = cv2.cvtColor(si_img, cv2.COLOR_RGB2GRAY)

    si_img = normalize(si_img)
    ss_img = normalize(ss_img)
    hs_img = normalize(hs_img)

    if si_img.shape[:2] != ss_img.shape[:2] or si_img.shape[:2] != hs_img.shape[:2]:
        raise ValueError(f"Dimensions don't correspond for {si_path}")
        
    for i, setting in enumerate(fusion_settings):
         if (setting == 'RGB'):
             img = cv2.cvtColor(si_img, cv2.COLOR_RGB2GRAY)
         elif (setting == 'R'):
             img = si_img[:,:, 0]
         elif (setting == 'G'):
             img = si_img[:,:, 1]
         elif (setting == 'B'):
             img = si_img[:,:, 2]
         elif (setting == '3D'):
             img = ss_img
         elif (setting == 'HS'):
             img = hs_img
         fused_image[:,:,i] = img
             

    tiff.imwrite(output_path, fused_image, photometric='rgb')
    print(f"Saved fusion for an img: {output_path}")

