# 🧠 Preprocessing Tools for Lip Reading Model Analysis

This repository contains all preprocessing scripts and tools used in our analysis of non-frontal-view lip reading, particularly focused on experiments involving **speech unit type**, **phonetic ambiguity**, and **video resolution**. It also includes utilities for running the **MonoNPHM** model to transform facial pose in the **LRS3** dataset.

## 📁 Contents

### ✅ Preprocessing Scripts
- `speech_unit_type/`  
  Scripts for grouping and labeling data based on speech unit type (e.g., words vs. sentences).
  
- `phonetic_ambiguity/`  
  Scripts to organize samples by phonetic similarity or ambiguity.
  
- `video_resolution/`  
  Tools for downscaling videos to simulate low-resolution inputs.

### 🧰 MonoNPHM Utilities
- Step 1: `extract_frames.py` and `extract_frames.slurm` if working on a shared cluster
  Shell script to extract video frames into images first before inputting it into the model
- Step 2: `preprocessing_step.slurm`
  Shell script of the preprocessing step that involves landmark detection, facial segmentation, and background matting
- Step 3: `stage1.slurm`
  Shell script of the stage 1 tracking phase of mononphm that uses model-based photometric 3D head tracking from the MonoNPHM framework to recreate dynamic facial motion from single-camera RGB videos
- Step 4: `stage2.slurm`
  Shell script of stage 2 tracking phase of mononphm that refine the tracking results by fine-tuning identity-specific features
- Step 5: `render_videos.slurm` that turns the meshes into synthetic videos using Trimesh and pyrender

### 🎯 Face Pose Angle Measurement
- `dataset_angle_count/`  
  Python scripts to count and log the face pose angles (yaw, pitch, roll) from video frames 

## 📦 Dependencies

Please install the following Python packages (via `pip` or `conda`):
- `numpy`
- `opencv-python`
- `trimesh`
- `pyrender`
- `matplotlib`
- `imageio`
- `scipy`

MonoNPHM requires:
- PyTorch (version based on MonoNPHM repo)
- GPU with CUDA
- OpenGL (`egl` or `osmesa` for headless rendering)

This repo assumes access to the following datasets:
- [LRS3](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html)
- [OuluVS2](https://www.oulu.fi/en/university/faculties-and-units/faculty-information-technology-and-electrical-engineering/center-machine-vision-and-signal-analysis/data-collections/ouluvs2)

## 🛠 How to Use

Most scripts in this repository can be run directly using the terminal. Depending on the file type:

- For **Python scripts**:
  ```bash
  python filename.py
  ```
  Example:
  ```bash
  python downscale0.5.py
  ```
- For shell scripts or SLURM job files:
  ```bash
  bash filename.sh
  ```
  or (for SLURM-managed clusters):
  ```bash
  sbatch filename.slurm
  ```
- For Jupyter notebooks, open with:
  ```bash
  jupyter notebook
  ```
  Then select and run the notebook from your browser.
    

## 📄 License
This project is for research and academic use only. Please refer to the original licenses of the LRS3 dataset and MonoNPHM model.

## 👥 Acknowledgments

- [MonoNPHM](https://github.com/YudongYao/MonoNPHM) for the 3D face mesh generation model.
- [LRS3 Dataset](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html) by the University of Oxford VGG group for large-scale visual speech recognition research.
- [OuluVS2 Dataset](https://www.oulu.fi/en/university/faculties-and-units/faculty-information-technology-and-electrical-engineering/center-machine-vision-and-signal-analysis/data-collections/ouluvs2) provided by the University of Oulu, for research on multi-view and multi-modal visual speech recognition.

