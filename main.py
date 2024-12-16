import os
import json
import sys
from src import prepare_yolo_annotations, copy_files, create_yaml_file
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from PIL import Image

# Chemins des fichiers et répertoires
json_file_path = "large_rock_dataset.json"
OUTPUT_DIR = "datasets"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
if not(os.path.isfile('data.yaml')): create_yaml_file(OUTPUT_DIR) # Creates YAML file if it does not exist
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
DATA_YAML = str(Path('data.yaml').resolve())  # Chemin absolu vers votre fichier YAML existant
IMAGE_DIR = "swissImage_50cm_patches/"


# Model settings
DROPOUT_PROB = 0.05
PERSPECTIVE_PROB = 0.001
MODEL_NAME = 'yolo11s.pt'
PROJ_NAME = f"rock_detection_dp-{DROPOUT_PROB}_pp-{PERSPECTIVE_PROB}_{MODEL_NAME.replace('.pt','')}"
MODEL_OUTPUT = 'runs/detect/' + PROJ_NAME

def main():
    # 1. Charger les données JSON
    print("Chargement des données JSON...")
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        dataset = data['dataset']
    print('Number of samples  :', len(dataset) )
    print(f"{len(dataset)} images trouvées dans le dataset.")
    

    # 2. Préparer les annotations au format YOLO
    print("Conversion des annotations en format YOLO...")
    annotations_dir = os.path.join(OUTPUT_DIR, "annotations")
    prepare_yolo_annotations(dataset, annotations_dir)

    # 3. Diviser les données en train/val
    print("Division des données en ensembles train/val...")
    image_files = [sample["file_name"].replace(".tif", "") for sample in dataset]
    train_files, val_files = train_test_split(image_files, train_size=0.8, random_state=42)

    copy_files(
        train_files,
        src_img=IMAGE_DIR,  # a modif
        src_ann=annotations_dir,      
        dest_img=f"{OUTPUT_DIR}/train/images",
        dest_ann=f"{OUTPUT_DIR}/train/labels",
    )

    copy_files(
        val_files,
        src_img=IMAGE_DIR,  # a modif
        src_ann=annotations_dir,       # Chemin des annotations
        dest_img=f"{OUTPUT_DIR}/val/images",
        dest_ann=f"{OUTPUT_DIR}/val/labels",
    )
    
     
    # 4. Entraîner YOLO
    print("Entraînement du modèle YOLO...")
    model = None                # Clear previous model from memory
    model = YOLO(MODEL_NAME).to('cuda')  # Utilisez un modèle YOLO pré-entraîné 
    model.train(data=DATA_YAML, time=1.7, 
                imgsz=640, batch=16,
                name=PROJ_NAME,
                dropout= DROPOUT_PROB, perspective=PERSPECTIVE_PROB)

    print("Pipeline terminé avec succès !")
    torch.cuda.empty_cache()    # Clear GPU memory once finished
    plt.imshow(Image.open(MODEL_OUTPUT+'/results.png'))
    plt.show()

if __name__ == "__main__":
    main()

