import os
import json
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from src import prepare_yolo_annotations, copy_files
from pathlib import Path

# Chemins des fichiers et répertoires
json_file_path = "large_rock_dataset.json"
OUTPUT_DIR = "datasets"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
DATA_YAML = str(Path('data.yaml').resolve())
IMAGE_DIR = "swissImage_50cm_patches/"

# Liste des hyperparamètres à tester
grid_params = {
    'lr0': [0.001, 0.01, 0.1],  # Taux d'apprentissage initial
    'momentum': [0.8, 0.9],     # Momentum
    'weight_decay': [0.0005, 0.005],  # Décroissance de poids
    'iou': [0.4, 0.5, 0.6]      # Seuil d'IoU pour considérer une détection correcte
}

def main():
    # 1. Charger les données JSON
    print("Chargement des données JSON...")
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        dataset = data['dataset']
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
        src_img=IMAGE_DIR,
        src_ann=annotations_dir,
        dest_img=f"{OUTPUT_DIR}/train/images",
        dest_ann=f"{OUTPUT_DIR}/train/labels",
    )

    copy_files(
        val_files,
        src_img=IMAGE_DIR,
        src_ann=annotations_dir,
        dest_img=f"{OUTPUT_DIR}/val/images",
        dest_ann=f"{OUTPUT_DIR}/val/labels",
    )
    

    print("Début du Grid Search pour trouver les meilleurs hyperparamètres...")
    best_model = None
    best_params = None
    #la fitness est une métrique globale utilisée pour évaluer la performance du modèle. Elle est souvent
    # définie comme une combinaison pondérée de plusieurs métriques clés pour l'entraînement d'un modèle de détection d'objets. 
    best_fitness = float('-inf')  # Maximiser la fitness 

    for weight_decay in grid_params['weight_decay']:
       for iou in grid_params['iou']:
           print(f"Entraînement avec weight_decay={weight_decay}, iou={iou}...")
           model = YOLO("yolov11n.pt").to('cuda')
                    
           # Entraînement avec les hyperparamètres actuels
           results = model.train(
                       batch=16,
                        data=DATA_YAML,
                        epochs=10,           # Fixer les epochs à modifier !
                        weight_decay=weight_decay,
                        save=False          # Désactiver la sauvegarde pour économiser de l'espace
                    )
                    
           # Évaluer les performances (fitness)
           fitness = results.results.fitness if hasattr(results, 'results') and hasattr(results.results, 'fitness') else 0
           print(f"Fitness obtenu : {fitness}")
                    
           # Sauvegarder les meilleurs résultats
           if fitness > best_fitness:
               best_fitness = fitness
               best_model = model
               best_params = {
                            'weight_decay': weight_decay,
                            'iou': iou
                        }

    print(f"Meilleurs hyperparamètres trouvés : {best_params}")
    print(f"Meilleure fitness : {best_fitness}")
    print("Pipeline terminé avec succès !")


if __name__ == "__main__":
    main()
