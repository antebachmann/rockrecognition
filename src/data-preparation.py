def prepare_yolo_annotations(dataset, output_dir):
    """
    Convert all annotations in the dataset to YOLO format.
    Args:
        dataset (list): List of samples with file names and annotations.
        output_dir (str): Directory to save YOLO annotation files.
    """
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

