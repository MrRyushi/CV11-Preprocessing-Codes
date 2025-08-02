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
