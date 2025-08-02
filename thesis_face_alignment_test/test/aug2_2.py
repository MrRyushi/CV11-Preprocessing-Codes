import trimesh
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dreifus.matrix import Intrinsics, Pose
from dreifus.camera import CameraCoordinateConvention

mica_pose = np.load('video3_frame0/00000_cam_params_opencv.npz')

c2w_pose = Pose(mica_pose['R'][0], mica_pose['t'][0],
                camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV)
c2w_pose = c2w_pose.invert().change_camera_coordinate_convention(
    new_camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL).invert()
rot = np.array(c2w_pose)[:3, :3]
trans = np.array(c2w_pose)[:3, 3]

# === Load mesh and transforms ===
mesh_nphm = trimesh.load("video3_frame0/mesh.ply", process=False)

s = np.squeeze(np.load("video3_frame0/scale.npy"))
R = np.squeeze(np.load("video3_frame0/rot.npy"))
t = np.squeeze(np.load("video3_frame0/trans.npy"))

anchors = None

sim_nphm1 = np.eye(4)
sim_nphm1[:3, :3] = 4*s*R
sim_nphm1[:3, 3] = t

# transform nphm mesh to FLAME coordinates and FLAME head pose
mesh_nphm.vertices = 1/s *  (mesh_nphm.vertices - t) @ R
mesh_nphm.vertices /= 4

# transform from FLAME coordinates to camera space
mesh_nphm.vertices = mesh_nphm.vertices @ rot.T + trans

sim_nphm2 = np.eye(4)
sim_nphm2[:3, :3] = rot
sim_nphm2[:3, 3] = trans

# also transform anchors in the same way
if anchors is not None:
    anchors = 1/s *  (anchors-t) @ R
    anchors /= 4
    anchors = anchors @ rot.T + trans


# OpenVC to OpenGL
mesh_nphm.vertices[:, 2] *= -1
mesh_nphm.vertices[:, 1] *= -1
sim = np.eye(4)
sim[:, 2] *= -1
sim[:, 1] *= - 1
similarity_transforms = sim @ similarity_transforms
if anchors is not None:
    anchors[:, 2] *= -1
    anchors[:, 1] *= -1

pl = pv.Plotter()
pl.add_points(points3d_dreifus[forground])
pl.add_mesh(mesh_flame)
if mesh_nphm is not None:
    pl.add_mesh(mesh_nphm, color='green')
pl.add_points(np.zeros([3]), color='red')
pl.show()

pl = pv.Plotter()
valid_lm_inds = valid_landmarks_dreifus[17:48]
pl.add_points(landmarks_flame[17:48, :][valid_lm_inds, :])
pl.add_points(landmarks3d_dreifus[17:48, :][valid_lm_inds, :], color='red')
for i in range(17, 48):
    if valid_landmarks_dreifus[i]:
        pl.add_mesh(pv.Line(landmarks_flame[i], landmarks3d_dreifus[i]))
pl.show()
