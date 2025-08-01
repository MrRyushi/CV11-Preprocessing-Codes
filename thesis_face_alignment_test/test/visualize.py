import trimesh
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# Load 3D mesh
mesh = trimesh.load("video2_frame0/mesh.ply", process=False)
vertices = mesh.vertices  # (N, 3)

# Load transformation parameters
trans = np.load("video2_frame0/trans.npy")
scale = np.load("video2_frame0/scale.npy").item()
rot = np.load("video2_frame0/rot.npy")

# Convert rotation to matrix
if rot.shape == (1, 3, 3):
    rot_matrix = rot[0]
elif rot.shape == (3, 3):
    rot_matrix = rot
elif rot.shape == (1, 3):
    rot_matrix = R.from_rotvec(rot[0]).as_matrix()
else:
    raise ValueError(f"Unexpected rot.npy shape: {rot.shape}")

# Apply rigid transformation
transformed = vertices @ rot_matrix.T + trans

# Project to 2D (weak perspective)
projected_2d = transformed[:, :2] * scale
projected_2d = projected_2d * 224  # assuming crop is 224x224


# ⬇️ Skip bbox scaling — image and crop are same resolution (224×224)

# Load image (should be 224x224)
image = cv2.imread("video2_frame0/00000.png")
h, w = image.shape[:2]
assert (w, h) == (224, 224), f"Expected 224×224 image, got {w}×{h}"

# Load lip indices
with open("video2_frame0/lip_indices_real.txt", "r") as f:
    lip_indices = [int(line.strip()) for line in f if line.strip().isdigit()]

# Print debug info
print("projected_2d min/max:", projected_2d.min(axis=0), projected_2d.max(axis=0))
print("First 10 lip indices:", lip_indices[:10])
print("First 10 projected points:", projected_2d[lip_indices[:10]])

# Draw red dots on lip landmarks
for idx in lip_indices:
    x, y = projected_2d[idx]
    if 0 <= x < w and 0 <= y < h:
        cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255), -1)

# Show or save
cv2.imshow("Lip Landmarks Overlay", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: Save overlay
cv2.imwrite("video2_frame0/lips_overlay_mononphm.png", image)
