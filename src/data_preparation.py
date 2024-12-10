import os
import shutil

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

