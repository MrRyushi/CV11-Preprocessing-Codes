import numpy as np

bbox_data = np.load("bboxes.npy", allow_pickle=True)
print("Type:", type(bbox_data))
print("Keys or shape:", getattr(bbox_data, 'keys', lambda: None)() if isinstance(bbox_data, dict) else bbox_data.shape)

# If it's a dictionary, try accessing the relevant frame
if isinstance(bbox_data, dict):
    print("Available keys (frame names):", list(bbox_data.keys())[:5])  # show a few keys
    print("Bounding box for video2_frame0 or 00000.png:", bbox_data.get('00000.png', 'Not found'))
else:
    print("Sample entry (first row):", bbox_data[0])
