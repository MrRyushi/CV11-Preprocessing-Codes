import numpy as np
import cv2

landmarks_data = np.load("00000.npy", allow_pickle=True).item()

# Try common keys (depends on the 3DDFA version)
print("Available keys:", landmarks_data.keys())

# For example, if it's in 'pts'
landmarks_2d = landmarks_data['pts']  # shape should be (68, 2)
print("Landmarks 2D shape:", landmarks_2d.shape)


# --- Load 3D mesh (MonoNPHM or 3DDFA output) ---
def load_vertices_from_obj(obj_path):
    vertices = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                vertices.append([x, y, z])
    return np.array(vertices)

lip_3d = np.loadtxt("mononphm_vertices.txt")  # shape: (N, 3)
print("Lip 3D shape:", lip_3d.shape)


# --- Dummy projection for visualization ---
def weak_perspective_projection(points3D, image):
    scale = 200  # arbitrary scale
    center = np.array([image.shape[1] // 2, image.shape[0] // 2])
    projected = points3D[:, :2] * scale + center
    return projected

# --- Load image background ---
image = cv2.imread("00000.png")
if image is None:
    raise ValueError("Could not load 00000.png. Make sure it's in the same folder.")

# --- Project 3D lip points to 2D image plane ---
projected_2d = weak_perspective_projection(lip_3d, image)
print("Projected 2D shape:", projected_2d.shape)

# --- Draw circles ---
for (x, y) in projected_2d.astype(int):
    cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

# --- Draw original 2D landmarks for comparison ---
for (x, y) in landmarks_2d.astype(int):
    cv2.circle(image, (x, y), 2, (255, 255, 0), -1)

# --- Save the result ---
cv2.imwrite("visualization_result.png", image)
print("✅ Saved visualization_result.png")