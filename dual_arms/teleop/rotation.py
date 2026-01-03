import time
import cv2
import math
import mediapipe as mp
from dual_arms.utils.one_euro_filter import OneEuroFilter
from importlib.resources import files
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

class HandOrientationTuner:
    def __init__(self):
        self.fps = 30
        

        model_path = str(files("dual_arms.teleop.models").joinpath("hand_landmarker.task"))
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.prev_quat = None
        self.alpha = 0.25

    def get_orientation(self):
        ret, frame = self.cap.read()
        if not ret: return None, None
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp = int(time.time() * 1000)

        result = self.landmarker.detect_for_video(mp_image, timestamp)
        smooth_quat = None

        if result.hand_landmarks:
            hand_o = result.hand_landmarks[0]
            hand = result.hand_world_landmarks[0]

            wrist_o = hand_o[0]

            wrist = np.array([hand[0].x, hand[0].y, hand[0].z])
            middle = np.array([hand[9].x, hand[9].y, hand[9].z])
            pinky = np.array([hand[17].x, hand[17].y, hand[17].z])

            vec_x = pinky - wrist
            vec_x = vec_x / np.linalg.norm(vec_x)

            # 2. Temp Vector (Wrist -> Pinky)
            vec_temp = middle - wrist

            # 3. Z-Axis (Normal): X cross Temp
            vec_z = np.cross(vec_x, vec_temp)
            nz = np.linalg.norm(vec_z)

            if nz < 1e-4:
                return None, frame  # or reuse last rotation

            vec_z /= nz

            # 4. Y-Axis (Ortho): Z cross X
            # This ensures Y is perfectly 90 degrees to X and Z
            vec_y = np.cross(vec_z, vec_x)
            vec_y = vec_y / np.linalg.norm(vec_y)

            # 5. Build Rotation Matrix
            # Stack them as columns: [x, y, z]
            rot_matrix = np.column_stack((vec_x, vec_y, vec_z))

            # 6. Convert to Quaternion [x, y, z, w]
            r = R.from_matrix(rot_matrix)
            quat = r.as_quat()

            # after quat = r.as_quat()

            if self.prev_quat is not None:
                if np.dot(self.prev_quat, quat) < 0:
                    quat = -quat

            if self.prev_quat is None:
                smooth_quat = quat
            else:
                # 2) slerp smoothing
                key_rots = R.from_quat([self.prev_quat, quat])
                slerp = Slerp([0, 1], key_rots)
                smooth_quat = slerp([self.alpha])[0].as_quat()

            self.prev_quat = smooth_quat
            r_smooth = R.from_quat(smooth_quat)
            rot_matrix = r_smooth.as_matrix()

            # use rot_matrix columns for drawing (vec_x, vec_y, vec_z)
            vec_x, vec_y, vec_z = rot_matrix[:,0], rot_matrix[:,1], rot_matrix[:,2]


            # --- VISUALIZATION (2D) ---
            origin = (int(hand_o[0].x * w), int(hand_o[0].y * h))
            scale = 150

            # Draw X (Red), Y (Green), Z (Blue)
            # We access the columns of the matrix for the axes
            cv2.line(frame, origin, (int(origin[0] + vec_x[0]*scale), int(origin[1] + vec_x[1]*scale)), (0, 0, 255), 3)
            cv2.line(frame, origin, (int(origin[0] + vec_y[0]*scale), int(origin[1] + vec_y[1]*scale)), (0, 255, 0), 3)
            cv2.line(frame, origin, (int(origin[0] + vec_z[0]*scale), int(origin[1] + vec_z[1]*scale)), (255, 0, 0), 3)

            # Display the Quaternion numbers
            text = f"Q: [{quat[0]:.2f}, {quat[1]:.2f}, {quat[2]:.2f}, {quat[3]:.2f}]"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        return smooth_quat, frame

    def close(self):
        self.cap.release()
        self.landmarker.close()

def main():
    tuner = HandOrientationTuner()
    while True:
        rotation, frame = tuner.get_orientation()
        
        cv2.imshow('Step 1: Raw Vectors', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tuner.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()