import os
import trimesh
import pyrender
import numpy as np
import cv2

# ✅ Force EGL for rendering
os.environ["PYOPENGL_PLATFORM"] = "egl"  # Or use 'osmesa' if EGL fails

# Base paths
stage2_dir = "../models/MonoNPHM/tracking_output/pretrained_mononphm_original/stage2"
tracking_input_dir = "../models/MonoNPHM/tracking_input"
output_base_dir = "../datasets/face_pose_transformed/lrs2_face_transformed"
os.makedirs(output_base_dir, exist_ok=True)

# Iterate through all sequence folders in stage2
for seq_name in sorted(os.listdir(stage2_dir)):
    seq_path = os.path.join(stage2_dir, seq_name)
    if not os.path.isdir(seq_path):
        continue  # Skip non-directory files

    output_dir = os.path.join(output_base_dir, seq_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get frame directories (e.g., 00000, 00001, ...)
    frame_dirs = sorted([f for f in os.listdir(seq_path) if f.isdigit()])
    frame_images = []

    # Process each frame
    for frame_dir in frame_dirs:
        try:
            frame_path = os.path.join(seq_path, frame_dir)
            print(f"Processing: {frame_path}")

            # Load mesh
            mesh_path = os.path.join(frame_path, "mesh.ply")
            if not os.path.exists(mesh_path):
                print(f"Mesh not found: {mesh_path}, skipping...")
                continue

            mesh = trimesh.load_mesh(mesh_path)

            # Load transformation parameters
            rot = np.load(os.path.join(frame_path, "rot.npy"))
            trans = np.load(os.path.join(frame_path, "trans.npy"))
            scale = np.load(os.path.join(frame_path, "scale.npy")).item()

            # Apply transformations
            mesh.apply_scale(scale)
            rot = rot.reshape(3, 3)
            trans = trans.reshape(3, 1)

            # Create (4x4) transformation matrix
            T = np.eye(4)
            T[:3, :3] = rot
            T[:3, 3] = trans.squeeze()
            mesh.apply_transform(T)

            # Rotate 90° for profile view
            R_side = trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])
            mesh.apply_transform(R_side)

            # ✅ Scale the mesh if it appears too small
            mesh.apply_scale(5.0)

            # Render the mesh
            scene = pyrender.Scene()
            scene.add(pyrender.Mesh.from_trimesh(mesh))

            # Add camera and lighting
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = [0, 0, 3.0]
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0)
            scene.add(camera, pose=camera_pose)

            light_color = np.ones(3)  # White light
            scene.add(pyrender.DirectionalLight(color=light_color, intensity=4.0), pose=np.eye(4))

            renderer = pyrender.OffscreenRenderer(800, 800)
            color, _ = renderer.render(scene)

            # Save frame temporarily
            frame_images.append(color)

            # Clean up
            renderer.delete()

        except Exception as e:
            print(f"Error processing frame {frame_dir} in {seq_name}: {e}")

    # Consolidate frames into a video
    if not frame_images:
        print(f"No frames found for {seq_name}, skipping video creation.")
        continue

    height, width, _ = frame_images[0].shape
    video_output_path = os.path.join(output_dir, f"{seq_name}.mp4")
    fps = 24
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    for frame in frame_images:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video_writer.release()
    print(f"Video saved to {video_output_path}")

    # Copy transcript file
    transcript_path = os.path.join(tracking_input_dir, seq_name, f"{seq_name}.txt")
    transcript_output_path = os.path.join(output_dir, f"{seq_name}.txt")
    if os.path.exists(transcript_path):
        os.system(f"cp {transcript_path} {transcript_output_path}")
        print(f"Transcript saved to {transcript_output_path}")
    else:
        print(f"Transcript not found: {transcript_path}")