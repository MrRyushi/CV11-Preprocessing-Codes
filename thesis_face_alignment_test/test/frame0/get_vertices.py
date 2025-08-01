import numpy as np
import cv2
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load mesh
mesh = trimesh.load('./mesh.ply', process=False)
vertices = mesh.vertices  # shape: (N, 3)

# From eval.py — landmark indices used in MonoNPHM
lm_inds = np.array([
    2212, 3060, 3485, 3384, 3386, 3389, 3418, 3395, 3414, 3598, 3637,
    3587, 3582, 3580, 3756, 2012, 730, 1984, 3157, 335, 3705, 3684,
    3851, 3863, 16, 2138, 571, 3553, 3561, 3501, 3526, 2748, 2792,
    3556, 1675, 1612, 2437, 2383, 2494, 3632, 2278, 2296, 3833, 1343,
    1034, 1175, 884, 829, 2715, 2813, 2774, 3543, 1657, 1696, 1579,
    1795, 1865, 3503, 2948, 2898, 2845, 2785, 3533, 1668, 1730, 1669,
    3509, 2786
])

# Extract landmark vertices
landmark_vertices = vertices[lm_inds]

# Print ranges (debugging)
print("X range:", landmark_vertices[:, 0].min(), landmark_vertices[:, 0].max())
print("Y range:", landmark_vertices[:, 1].min(), landmark_vertices[:, 1].max())
print("Z range:", landmark_vertices[:, 2].min(), landmark_vertices[:, 2].max())

# === STEP 1: 3D Face Visualization ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(landmark_vertices[:, 0], landmark_vertices[:, 1], landmark_vertices[:, 2], c='red', s=10)
ax.set_title("Raw 3D Landmarks from MonoNPHM")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

# === STEP 2: 2D Projection ===
# Drop Z
landmarks_2d = landmark_vertices[:, :2]  # shape (68, 2)

# === STEP 3: Optional 90-degree rotation ===
rotate_90 = True
if rotate_90:
    theta = np.deg2rad(90)
    Rz = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    landmarks_2d = landmarks_2d @ Rz.T

# === STEP 4: Normalize to image ===
img = cv2.imread("00000.png")  # Change this to your image file
h, w = img.shape[:2]

# Center and scale
center = landmarks_2d.mean(axis=0)
landmarks_centered = landmarks_2d - center
max_range = np.max(np.abs(landmarks_centered))  # uniform scale
scale = 0.45 * min(w, h) / max_range
landmarks_img = landmarks_centered * scale + np.array([w / 2, h / 2])

# Optional: flip Y to match OpenCV convention
flip_y = True
if flip_y:
    landmarks_img[:, 1] = h - landmarks_img[:, 1]

# === STEP 5: Overlay on image ===
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.scatter(landmarks_img[:, 0], landmarks_img[:, 1], c='red', s=10, label='MonoNPHM Landmarks')
plt.legend()
plt.title("Projected MonoNPHM Landmarks on Image")
plt.axis("off")
plt.show()
