import time
import cv2
import math
import mediapipe as mp
from dual_arms.utils.one_euro_filter import OneEuroFilter
from importlib.resources import files
import numpy as np


class HandTracker:
    def __init__(self):
        self.SCALE_X = 1.0
        self.SCALE_Y = 1.0
        self.SCALE_Z = 1.0

        self.ROBOT_HOME = np.array([0.4, 0.2, 0.0])

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

            wp4, wp6 = hand[4], hand[6]
            wp17, wp20 = worldHand[17], worldHand[20]

            
            closed_val = (math.sqrt((wp4.x-wp6.x)**2 + ((wp4.y-wp6.y)**2)) + math.sqrt((wp17.x-wp20.x)**2 + ((wp17.y-wp20.y)**2))) * 100

            closed = 0.0 if closed_val < 7.5 else 4

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
            cv2.putText(frame, f"{closed:.2f} x: {filt_x:.2f} y: {filt_y:.2f} Z: {filt_z:.2f}", (cx+15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        return target_pos, closed, frame

    def close(self):
        self.cap.release()
        self.landmarker.close()

        