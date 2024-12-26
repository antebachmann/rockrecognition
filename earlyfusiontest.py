import os
from PIL import Image
from tifffile import tifffile as tiff
from src import early_fusion

# Chemins des dossiers source

src_SI = "swissImage_50cm_patches/"
src_SS = "swissSURFACE3D_patches/"
src_HS = "swissSURFACE3D_hillshade_patches/"

# Dossier de sortie pour les images combinées
output_dir = "combined_images"
os.makedirs(output_dir, exist_ok=True)

# Parcourir les fichiers et réaliser l'early fusion
file_names = os.listdir(src_SI)
for file_name in file_names:
    try:
        si_path = os.path.join(src_SI, file_name)
        ss_path = os.path.join(src_SS, file_name)
        hs_path = os.path.join(src_HS, file_name)
        output_path = os.path.join(output_dir, file_name)

        early_fusion(si_path, ss_path, hs_path, output_path)
    except Exception as e:
        print(f"Erreur pour {file_name}: {e}")


# Charger une image fusionnée
fused_image = tiff.imread('combined_images/2781_1141_3_3.tif')
print("Dimensions de l'image fusionnée :", fused_image.shape)  # (hauteur, largeur, 5)

# Afficher les différents canaux
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
for i in range(fused_image.shape[2]):
    plt.subplot(1, 5, i+1)
    plt.imshow(fused_image[:, :, i], cmap='gray')
    plt.title(f"Canal {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.show()


