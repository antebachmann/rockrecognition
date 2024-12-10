import os
import json
import sys
import os
from src import prepare_yolo_annotations, copy_files
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

# Chemins des fichiers et répertoires
json_file_path = "C:/Users/asus/Desktop/Rocks detection project/LargeRocksDetectionDataset/LargeRocksDetectionDataset/large_rock_dataset.json"
OUTPUT_DIR = "C:/Users/asus/Desktop/rockrecognition/output"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
DATA_YAML = "C:/Users/asus/Desktop/rockrecognition/models/first_try_model.yaml"  # Chemin vers votre fichier YAML existant
IMAGE_DIR = "C:/Users/asus/Desktop/Rocks detection project/LargeRocksDetectionDataset/LargeRocksDetectionDataset/swissImage_50cm_patches/"

def main():
    # 1. Charger les données JSON
    print("Chargement des données JSON...")
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        dataset =data['dataset']
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
    model = YOLO("yolov8n.pt")  # Utilisez un modèle YOLO pré-entraîné 
    model.train(data=DATA_YAML, epochs=50, imgsz=640, batch=16, name="rock_detection")

    print("Pipeline terminé avec succès !")

if __name__ == "__main__":
    main()

