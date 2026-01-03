import time
import cv2
import math
import mediapipe as mp
from dual_arms.utils.one_euro_filter import OneEuroFilter
from importlib.resources import files
import numpy as np


class HandTracker:
    def __init__(self):
        self.SCALE_X = 0
        self.SCALE_Y = 0
        self.SCALE_Z = 0

        self.ROBOT_HOME = np.array([0.0, 0.0, 0.0])

        self.fps = 30
        self.filters = (
            OneEuroFilter(freq=self.fps, mincutoff=0.15, beta=0.2),
            OneEuroFilter(freq=self.fps, mincutoff=0.15, beta=0.2),
            OneEuroFilter(freq=self.fps, mincutoff=0.01, beta=0.03)
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

    def euclidean_distance(self, point1, point2):
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def is_hand_closed(self, hand):
        wrist = hand[0]

        wp8, wp6 = hand[8], hand[6]
        wp12, wp10 = hand[12], hand[10]
        wp16, wp14 = hand[16], hand[14]
        wp20, wp18 = hand[20], hand[18]

        dist_tip1 = self.euclidean_distance(wp8, wrist)
        dist_pip1 = self.euclidean_distance(wp6, wrist)

        dist_tip2 = self.euclidean_distance(wp12, wrist)
        dist_pip2 = self.euclidean_distance(wp10, wrist)

        dist_tip3 = self.euclidean_distance(wp16, wrist)
        dist_pip3 = self.euclidean_distance(wp14, wrist)

        dist_tip4 = self.euclidean_distance(wp20, wrist)
        dist_pip4 = self.euclidean_distance(wp18, wrist)

        index_closed = dist_tip1 < dist_pip1
        middle_closed = dist_tip2 < dist_pip2
        ring_closed = dist_tip3 < dist_pip3
        pinky_closed = dist_tip4 < dist_pip4

        hand_closed = index_closed or middle_closed or ring_closed or pinky_closed

        return hand_closed


    def get_target(self):
        ret, frame = self.cap.read()

        if not ret:
            return None, None
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp = int(time.time() * 1000)

        result = self.landmarker.detect_for_video(mp_image, timestamp)
        target_pos = None
        closed = 0.037

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            worldHand = result.hand_world_landmarks[0]

            raw_x, raw_y = hand[0].x, hand[0].y


            # hand closing calculation
            isClosed = self.is_hand_closed(worldHand)

            closed = 0 if isClosed else 10


            # z value
            p0, p9 = hand[0], hand[9]
            dist_vert = math.sqrt(((p0.x-p9.x)*w)**2 + ((p0.y-p9.y)*h)**2)
            z_vert = (1000.0 / dist_vert) if dist_vert > 0 else 9999

            p5, p17 = hand[5], hand[17]
            dist_horiz = math.sqrt(((p5.x-p17.x)*w)**2 + ((p5.y-p17.y)*h)**2)
            z_horiz = (668.0 / dist_horiz) if dist_horiz > 0 else 9999

            raw_z = min(z_vert, z_horiz)
            norm_z = max(0.0, min(1.0, (raw_z - 5) / (10 - 5)))

            t = time.time()
            filt_x = self.filters[0](raw_x, timestamp=t)
            filt_y = self.filters[1](raw_y, timestamp=t)
            filt_z = self.filters[2](norm_z, timestamp=t)

            r_y = (filt_x - 1.0) * self.SCALE_Y  
            
            # Invert Y: 0.0 (Top) -> +Z (Up)
            r_z = (0.5 - filt_y) * self.SCALE_Z
            
            # Depth Z: 0.0 (Far) -> -X (Back)
            r_x = -(filt_z - 0.5) * self.SCALE_X

            target_pos = self.ROBOT_HOME + np.array([r_y, r_x, r_z])

            # Draw Debug
            cx, cy = int(filt_x * w), int(filt_y * h)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
            cv2.putText(frame, f"{isClosed} x: {filt_x:.2f} y: {filt_y:.2f} Z: {filt_z:.2f}", (cx+15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        return target_pos, closed, frame

    def close(self):
        self.cap.release()
        self.landmarker.close()

        