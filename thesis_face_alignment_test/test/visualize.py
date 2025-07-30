import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# --- Load 3D mesh (MonoNPHM) ---
vertices = np.loadtxt("mononphm_lip_vertices.txt")  # shape: (N, 3)

# --- Load pose parameters from MonoNPHM ---
rot = np.load("rot.npy")         # shape (3,)
trans = np.load("trans.npy")     # shape (3,)
scale = np.load("scale.npy")     # shape (1,) or scalar

# Convert rotation vector to rotation matrix
rot_matrix = rot.squeeze()  # Now shape (3, 3)
trans = trans.squeeze()          # (3,)
scale = float(scale)   

# Apply transformation
transformed = (vertices @ rot_matrix.T) * scale + trans  # shape (N, 3)

# --- Load original image ---
image = cv2.imread("00000.png")
if image is None:
    raise ValueError("Could not load image.")

# Project to 2D using weak perspective (ignore Z, center in image)
image_center = np.array([image.shape[1] // 2, image.shape[0] // 2])
projected_2d = transformed[:, :2] + image_center  # shape (N, 2)

# --- Load 2D landmarks from 3DDFA ---
landmarks_data = np.load("00000.npy", allow_pickle=True).item()
landmarks_2d = landmarks_data['ldm68']  # shape (68, 2)

# --- Draw MonoNPHM projected vertices in red ---
for (x, y) in projected_2d.astype(int):
    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)


# --- Draw 3DDFA 2D landmarks in cyan ---
# for (x, y) in landmarks_2d.astype(int):
#     if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
#         cv2.circle(image, (x, y), 2, (255, 255, 0), -1)

# --- Save ---
cv2.imwrite("visualization_result.png", image)
print("✅ Saved visualization_result.png")
