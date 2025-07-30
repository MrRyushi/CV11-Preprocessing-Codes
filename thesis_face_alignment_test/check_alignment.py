import numpy as np
from scipy.spatial import procrustes
from sklearn.metrics import mean_squared_error

# Load vertex coordinates from your .txt files
verts_3ddfa = np.loadtxt("3ddfav3_vertices.txt")
verts_mononphm = np.loadtxt("mononphm_vertices.txt")

# Check shapes
print("3DDFA shape:", verts_3ddfa.shape)
print("MonoNPHM shape:", verts_mononphm.shape)

# Optional: downsample if one has more vertices
if verts_3ddfa.shape != verts_mononphm.shape:
    min_len = min(len(verts_3ddfa), len(verts_mononphm))
    verts_3ddfa = verts_3ddfa[:min_len]
    verts_mononphm = verts_mononphm[:min_len]

# Align using Procrustes
mtx1, mtx2, disparity = procrustes(verts_3ddfa, verts_mononphm)

# Compute MSE
mse = mean_squared_error(mtx1, mtx2)

print("Procrustes Disparity:", disparity)
print("Mean Squared Error (MSE):", mse)
