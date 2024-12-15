import os
import shutil
from PIL import Image
import numpy as np
from tifffile import tifffile as tiff


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
                width_rel = 10 / 640
                height_rel = 10 / 640
                f.write(f"0 {x_rel} {y_rel} {width_rel} {height_rel}\n")



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


def early_fusion(si_path, ss_path, hs_path, output_path):
    si_img = tiff.imread(si_path)  # Image RGB (3 canaux)
    ss_img = tiff.imread(ss_path)  # Image Surface 3D (1 canal)
    hs_img = tiff.imread(hs_path)  # Image Hillshade (1 canal)

    si_img = normalize(si_img)
    ss_img = normalize(ss_img)
    hs_img = normalize(hs_img)

    if si_img.shape[:2] != ss_img.shape[:2] or si_img.shape[:2] != hs_img.shape[:2]:
        raise ValueError(f"Les dimensions ne correspondent pas pour {si_path}")

    ss_img = np.expand_dims(ss_img, axis=2)  
    hs_img = np.expand_dims(hs_img, axis=2)
    fused_image = np.dstack((si_img, ss_img, hs_img))  

    tiff.imwrite(output_path, fused_image)
    print(f"saved fusion for an img: {output_path}")

