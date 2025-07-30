import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import procrustes
from sklearn.metrics import mean_squared_error

# === STEP 1: Load vertices from 3DDFA-V3 .obj file ===
def load_obj_vertices(obj_path):
    vertices = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):  # Only vertex lines
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                vertices.append([x, y, z])
    return np.array(vertices)

# Paths to your files
obj_3ddfa_path = "00017/00017_pcaTex.obj"  # or extractTex.obj
mononphm_txt_path = "mononphm_vertices.txt"           # exported from Blender

# Load both sets of vertices
verts_3ddfa = load_obj_vertices(obj_3ddfa_path)
print("Loaded 3DDFA-V3 vertices:", verts_3ddfa.shape)

verts_mononphm = np.loadtxt(mononphm_txt_path)
print("Loaded MonoNPHM vertices:", verts_mononphm.shape)

# === STEP 2: Find closest MonoNPHM vertex for each 3DDFA point ===
nn = NearestNeighbors(n_neighbors=1).fit(verts_mononphm)
distances, indices = nn.kneighbors(verts_3ddfa)

# Get the matched vertices
matched_mononphm = verts_mononphm[indices[:, 0]]

# === STEP 3: Procrustes Alignment and Error Analysis ===
mtx1, mtx2, disparity = procrustes(verts_3ddfa, matched_mononphm)
mse = mean_squared_error(mtx1, mtx2)

# === STEP 4: Report ===
print("\n=== COMPARISON RESULTS ===")
print("Number of Vertices Compared:", len(verts_3ddfa))
print("Procrustes Disparity:", disparity)
print("Mean Squared Error (MSE):", mse)
