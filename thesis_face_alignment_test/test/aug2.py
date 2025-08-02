import trimesh
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.spatial import procrustes
import matplotlib.pyplot as plt

# === Load MonoNPHM Mesh and Parameters ===
mesh = trimesh.load("video3_frame0/mesh.ply", process=False)
vertices = mesh.vertices.copy()

rot_raw = np.load("video3_frame0/rot.npy")
trans = np.load("video3_frame0/trans.npy")
scale = np.load("video3_frame0/scale.npy").item()

if rot_raw.shape == (1, 3, 3):
    rot_matrix = rot_raw[0]
elif rot_raw.shape == (3, 3):
    rot_matrix = rot_raw
elif rot_raw.shape == (1, 3):
    rot_matrix = R.from_rotvec(rot_raw[0]).as_matrix()
else:
    raise ValueError("Unexpected rot.npy shape")

# === Step 1: Undo MonoNPHM transform (normalize) ===
vertices = 1 / scale * (vertices - trans) @ rot_matrix
vertices /= 4  # Normalize like eval.py

# === Step 2: Apply alignment (will compute from landmarks) — placeholder here ===

# === Step 3: Flip to OpenGL Coordinate System ===
vertices[:, 1] *= -1
vertices[:, 2] *= -1


# === Project to 2D (weak perspective) ===
scale_factor = 600  # used just for visualization
projected_2d = vertices[:, :2] * scale * scale_factor

# === Load 3DDFA 2D Landmarks ===
landmarks_data = np.load("video3_frame0/00000.npy", allow_pickle=True).item()
landmarks_2d = landmarks_data['ldm68']  # shape: (68, 2)

# === Image dimensions for centering ===
image = cv2.imread("video3_frame0/00000.png")
h, w = image.shape[:2]

# === Center projection ===
projected_2d[:, 0] += w // 2
projected_2d[:, 1] += h // 2

# === Landmark indices from MonoNPHM (eval.py) ===
lm_inds = np.array([
    2212, 3060, 3485, 3384, 3386, 3389, 3418, 3395, 3414, 3598, 3637,
    3587, 3582, 3580, 3756, 2012, 730, 1984, 3157, 335, 3705, 3684,
    3851, 3863, 16, 2138, 571, 3553, 3561, 3501, 3526, 2748, 2792,
    3556, 1675, 1612, 2437, 2383, 2494, 3632, 2278, 2296, 3833, 1343,
    1034, 1175, 884, 829, 2715, 2813, 2774, 3543, 1657, 1696, 1579,
    1795, 1865, 3503, 2948, 2898, 2845, 2785, 3533, 1668, 1730, 1669,
    3509, 2786
])

# === Visualize lm_inds on the transformed 3D mesh ===
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

landmarks_3d = vertices[lm_inds]  # shape (68, 3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot all vertices for reference (optional, comment out if too dense)
# ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='gray', s=0.5, alpha=0.3)

# Plot just the landmarks
ax.scatter(landmarks_3d[:, 0], landmarks_3d[:, 1], landmarks_3d[:, 2], c='red', s=10)
for i, (x, y, z) in enumerate(landmarks_3d):
    ax.text(x, y, z, str(i), color='black', fontsize=6)

ax.set_title("MonoNPHM Landmark Indices (3D)")
ax.set_box_aspect([1, 1, 1])
plt.show()


projected_landmarks = projected_2d[lm_inds]  # (68, 2)

# === Step 4: Align MonoNPHM landmarks to 3DDFA landmarks ===
def compute_similarity_transform(X, Y):
    X_mean = X.mean(0)
    Y_mean = Y.mean(0)
    Xc = X - X_mean
    Yc = Y - Y_mean
    U, _, Vt = np.linalg.svd(Xc.T @ Yc)
    R_align = U @ Vt
    if np.linalg.det(R_align) < 0:
        U[:, -1] *= -1
        R_align = U @ Vt
    s = np.trace(Yc.T @ Xc @ R_align.T) / np.sum(Xc**2)
    t = Y_mean - s * X_mean @ R_align
    return s, R_align, t

s_align, R_align, t_align = compute_similarity_transform(projected_landmarks, landmarks_2d)

# Apply to all projected points
projected_2d_aligned = projected_2d @ R_align.T * s_align + t_align
aligned_landmarks = projected_2d_aligned[lm_inds]

# === Scatter Plot: Alignment Result ===
plt.figure()
plt.scatter(landmarks_2d[:, 0], landmarks_2d[:, 1], c='blue', s=10, label="3DDFA")
plt.scatter(aligned_landmarks[:, 0], aligned_landmarks[:, 1], c='red', s=10, label="MonoNPHM (aligned)")
plt.gca().invert_yaxis()
plt.axis("equal")
plt.title("2D Landmark Alignment: MonoNPHM to 3DDFA")
plt.legend()
plt.show()

# === Overlay red dots on image ===
image_copy = image.copy()
for (x, y) in aligned_landmarks.astype(int):
    if 0 <= x < w and 0 <= y < h:
        cv2.circle(image_copy, (x, y), 2, (0, 0, 255), -1)

#Optional: overlay 3DDFA (blue)
for (x, y) in landmarks_2d.astype(int):
    cv2.circle(image_copy, (x, y), 2, (255, 255, 0), -1)

cv2.imshow("Overlay: Aligned MonoNPHM Landmarks", image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
