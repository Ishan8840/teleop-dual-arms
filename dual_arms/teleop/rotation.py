import time
import cv2
import math
import mediapipe as mp
from dual_arms.utils.one_euro_filter import OneEuroFilter
from importlib.resources import files
import numpy as np

class HandOrientationTuner:
    def __init__(self):
        self.fps = 30
        
        # ONLY Rotation Filters retained
        self.filters_rot = (
            OneEuroFilter(freq=self.fps, mincutoff=0.01, beta=0), # Roll
            OneEuroFilter(freq=self.fps, mincutoff=0.01, beta=0.0), # Pitch
            OneEuroFilter(freq=self.fps, mincutoff=0.01, beta=0.0)  # Yaw
        )

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

    def calculate_rotation(self, world_landmarks):
        """
        Calculates rotation matrix from landmarks and converts to Euler angles.
        """
        p0 = np.array([world_landmarks[0].x, world_landmarks[0].y, world_landmarks[0].z])
        p9 = np.array([world_landmarks[9].x, world_landmarks[9].y, world_landmarks[9].z])
        p5 = np.array([world_landmarks[5].x, world_landmarks[5].y, world_landmarks[5].z])
        p17 = np.array([world_landmarks[17].x, world_landmarks[17].y, world_landmarks[17].z])

        # 1. Forward Vector (Z-axis): Wrist -> Middle Finger
        vec_forward = p9 - p0
        vec_forward /= np.linalg.norm(vec_forward)

        # 2. Across Vector (X-axis approx): Pinky -> Index
        vec_across = p5 - p17
        vec_across /= np.linalg.norm(vec_across)

        # 3. Normal Vector (Y-axis): Up out of palm
        vec_normal = np.cross(vec_forward, vec_across)
        vec_normal /= np.linalg.norm(vec_normal)

        # 4. Re-orthogonalize Across Vector
        vec_across = np.cross(vec_normal, vec_forward)
        vec_across /= np.linalg.norm(vec_across)

        # 5. Rotation Matrix [Across, Normal, Forward] -> [x, y, z]
        R = np.column_stack((vec_across, vec_normal, vec_forward))

        # 6. Euler Angles (ZYX convention)
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0
            
        return roll, pitch, yaw

    def draw_orientation(self, frame, landmarks):
        h, w, _ = frame.shape
        def get_pt(idx):
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h, lm.z * w]) 

        p0 = get_pt(0)   # Wrist
        p5 = get_pt(5)   # Index
        p9 = get_pt(9)   # Middle
        p17 = get_pt(17) # Pinky

        # Re-calculate vectors in pixel space for drawing
        vec_z = p9 - p0
        vec_z = vec_z / np.linalg.norm(vec_z) * 80 
        vec_x = p5 - p17
        vec_x = vec_x / np.linalg.norm(vec_x) * 80
        vec_y = np.cross(vec_z, vec_x)
        vec_y = vec_y / np.linalg.norm(vec_y) * 80

        origin = (int(p0[0]), int(p0[1]))

        # Blue: Forward (Z)
        cv2.line(frame, origin, (int(p0[0]+vec_z[0]), int(p0[1]+vec_z[1])), (255, 0, 0), 3) 
        # Red: Side (X)
        cv2.line(frame, origin, (int(p0[0]+vec_x[0]), int(p0[1]+vec_x[1])), (0, 0, 255), 3)
        # Green: Up (Y)
        cv2.line(frame, origin, (int(p0[0]+vec_y[0]), int(p0[1]+vec_y[1])), (0, 255, 0), 3)

    def get_orientation(self):
        ret, frame = self.cap.read()
        if not ret: return None, None
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp = int(time.time() * 1000)

        result = self.landmarker.detect_for_video(mp_image, timestamp)
        rotation_rpy = None

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            worldHand = result.hand_world_landmarks[0]

            # 1. VISUALIZE
            self.draw_orientation(frame, hand)

            # 2. CALCULATE
            raw_roll, raw_pitch, raw_yaw = self.calculate_rotation(worldHand)

            # 3. FILTER
            t = time.time()
            filt_roll = self.filters_rot[0](raw_roll, timestamp=t)
            filt_pitch = self.filters_rot[1](raw_pitch, timestamp=t)
            filt_yaw = self.filters_rot[2](raw_yaw, timestamp=t)
            
            rotation_rpy = np.array([filt_roll, filt_pitch, filt_yaw])

            # Display values on screen
            cx, cy = int(hand[0].x * w), int(hand[0].y * h)
            text = f"Roll: {filt_roll:.2f} Pitch: {filt_pitch:.2f} Yaw: {filt_yaw:.2f}"
            cv2.putText(frame, text, (cx - 100, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return rotation_rpy, frame

    def close(self):
        self.cap.release()
        self.landmarker.close()

def main():
    tuner = HandOrientationTuner()
    while True:
        rotation, frame = tuner.get_orientation()
        
        if frame is not None:
            cv2.imshow("Orientation Tuner", frame)
            
        if rotation is not None:
            # Print to console for easy reading
            print(f"R: {rotation[0]:.2f} | P: {rotation[1]:.2f} | Y: {rotation[2]:.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tuner.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()