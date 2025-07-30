import numpy as np
import trimesh

# Load binary .ply mesh file using trimesh
mesh = trimesh.load("mesh.ply", process=False)

# Extract vertex positions (N x 3)
full_3d = mesh.vertices  # shape: (96391, 3)

# Load your lip region vertex indices
lip_indices = np.loadtxt("lip_indices.txt", dtype=int)

# Extract just the lip region
lip_3d = full_3d[lip_indices]

# Save if you want
np.savetxt("mononphm_lip_vertices.txt", lip_3d)

print("✅ Extracted lip vertices:", lip_3d.shape)
