import trimesh
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# Load 3D mesh
mesh = trimesh.load("video3_frame0/mesh.ply", process=False)
vertices = mesh.vertices 
# Load 2D landmarks from 3DDFA-V3
landmarks_data = np.load("video3_frame0/00000.npy", allow_pickle=True).item()
landmarks_2d = landmarks_data['ldm68']  
print("Landmarks 2D shape:", landmarks_2d.shape)


# Load MonoNPHM transformation parameters
trans = np.load("video3_frame0/trans.npy")  # (3,)
scale = np.load("video3_frame0/scale.npy").item()  # scalar
rot = np.load("video3_frame0/rot.npy")

# Handle rotation matrix
if rot.shape == (1, 3, 3):
    rot_matrix = rot[0]
elif rot.shape == (3, 3):
    rot_matrix = rot
elif rot.shape == (1, 3):
    rot_matrix = R.from_rotvec(rot[0]).as_matrix()
else:
    raise ValueError("Unexpected rot.npy shape:", rot.shape)

# Apply rotation and translation
transformed = vertices @ rot_matrix.T + trans

# Custom scale factor to enlarge the face (increase if face still too small)
scale_factor = 600  
projected_2d = transformed[:, :2] * scale * scale_factor
projected_2d[:, 1] *= -1  

# Load image to get dimensions
image = cv2.imread("video3_frame0/00000.png")
h, w = image.shape[:2]

# Center face in the middle of the image
projected_2d[:, 0] += w  
projected_2d[:, 1] += h // 2.5

# Use MonoNPHM's landmark indices
lm_inds = np.array(
    [2212, 3060, 3485, 3384, 3386, 3389, 3418, 3395, 3414, 3598, 3637,
     3587, 3582, 3580, 3756, 2012, 730, 1984, 3157, 335, 3705, 3684,
     3851, 3863, 16, 2138, 571, 3553, 3561, 3501, 3526, 2748, 2792,
     3556, 1675, 1612, 2437, 2383, 2494, 3632, 2278, 2296, 3833, 1343,
     1034, 1175, 884, 829, 2715, 2813, 2774, 3543, 1657, 1696, 1579,
     1795, 1865, 3503, 2948, 2898, 2845, 2785, 3533, 1668, 1730, 1669,
     3509, 2786]
)

# Extract projected landmarks
projected_landmarks = projected_2d[lm_inds]

# ---- Visualization 1: Isolated scatter (no image) ----
plt.figure()
plt.scatter(projected_landmarks[:, 0], projected_landmarks[:, 1], c='red', s=10)
plt.gca().invert_yaxis()
plt.axis("equal")
plt.title("MonoNPHM Projected Landmarks (Scaled and Centered)")
plt.show()

# # --- Draw original 2D landmarks for comparison ---
# for (x, y) in landmarks_2d.astype(int):
#     cv2.circle(image, (x, y), 2, (255, 255, 0), -1)

# ---- Visualization 2: Overlay on image ----
for (x, y) in projected_landmarks.astype(int):
    if 0 <= x < w and 0 <= y < h:
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

# Save or show image
# cv2.imwrite("video3_frame0/overlay_result.png", image)
cv2.imshow("Overlay", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
