import os
from PIL import Image
from tifffile import tifffile as tiff
from src import early_fusion

# Paths to dataset sources

src_SI = "swissImage_50cm_patches/"
src_SS = "swissSURFACE3D_patches/"
src_HS = "swissSURFACE3D_hillshade_patches/"

# Output dir is relative to the setting
setting = "RGB"
output_dir = setting + " images"
os.makedirs(output_dir, exist_ok=True)

# Get every file and perform early fusion
file_names = os.listdir(src_SI)
for file_name in file_names:
    try:
        si_path = os.path.join(src_SI, file_name)
        ss_path = os.path.join(src_SS, file_name)
        hs_path = os.path.join(src_HS, file_name)
        output_path = os.path.join(output_dir, file_name)

        early_fusion(si_path, ss_path, hs_path, output_path, setting)
    except Exception as e:
        print(f"Error for {file_name}: {e}")


# Charger une image fusionnée
fused_image = tiff.imread(output_dir + '/2781_1141_3_3.tif')
print("Dimensions of fused image :", fused_image.shape)  # (hauteur, largeur, 5)

# Afficher les différents canaux
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
for i in range(fused_image.shape[2]):
    plt.subplot(1, 5, i+1)
    plt.imshow(fused_image[:, :, i], cmap='gray')
    plt.title(f"Channel {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.show()


