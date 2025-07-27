import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh # Open face mesh detector
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils # For displaying whole face mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1) # Drawing specifications

# Open webcam
cap = cv2.VideoCapture(0)

# While webcam is open
while cap.isOpened():
    
    success, image = cap.read() # Read image from webcam
    start = time.time() # Start timing how long the algorithm takes

    # Flip image horizontally for a later selfie-view display (i.e. not mirrored/flipped)
    # Convert color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # Program can only read from the image for improving performance
    image.flags.writeable = False

    # Get result (returns normalized values)
    results = face_mesh.process(image)

    # Can write on image (i.e. display text, etc.)
    image.flags.writeable = True

    # Convert color space back from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Image height, width, and number of channels for scaling values with image dimensions
    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    # If there are detections
    if results.multi_face_landmarks:
        # Run through all landmarks detected in the image
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                # Indexes for e.g. nose, ears, mouth, eyes
                if idx in [33, 263, 1, 61, 291, 199, 4]:
                    if idx == 4:
                        # Set nose2d and nose3d to exact values detected
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000) # Scale out values for 3d nose

                    # Coordinates of landmark (normalized values)
                    # Scale with height and width of image (i.e. convert back to image space)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    face_2d.append([x, y]) # Get 2D coordinates
                    face_3d.append([x, y, lm.z]) # Get 3D coordinates

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # Set up camera matrix
            focal_length = 1 * img_w

            # Intrinsic parameters of camera
            cam_matrix = np.array([[focal_length, 0, img_h / 2], 
                                   [0, focal_length, img_w / 2], 
                                   [0, 0, 1]])
            
            # Distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            # rot_vec = how much the points are rotated in the image
            # trans_vec = how much the points are translated around
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles for all axes (normalized)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # See where head is tilting
            # 90 DEGREES (SIDE)
            if y < -24 or y > 12:
                text = "side"
            # 60 DEGREES
            elif y < -21 or y > 11:
                text = "60"
            # 45 DEGREES
            elif y < -14 or y > 9:
                text = "45"
            # 30 DEGREES
            elif y < -11 or y > 8:
                text = "30"
            else:
                text = "front"

            # Display nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
            cv2.line(image, p1, p2, (255, 0, 0), 3)

            # Add text to image
            cv2.putText(image, str(text), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        end = time.time()
        total_time = end - start

        fps = 1 / total_time
        print("FPS: ", fps)

        cv2.putText(image, f'FPS: {int(fps)}' , (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        # Draw landmarks with drawing utilities set up
        mp_drawing.draw_landmarks(
            image=image, 
            landmark_list=face_landmarks, 
            connections=mp_face_mesh.FACEMESH_CONTOURS, 
            landmark_drawing_spec=drawing_spec, 
            connection_drawing_spec=drawing_spec
        )

    cv2.imshow("Head Pose Estimation", image)

    # `esc` or `q` key to terminate program
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release webcam
cap.release()
