import time
import cv2
import math
import mediapipe as mp
from dual_arms.utils.one_euro_filter import OneEuroFilter
from importlib.resources import files

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

FPS = 30
MIN_CUTOFF = 0.5
BETA_XY = 0.1
BETA_Z = 0.0005  
MIN_CUTOFF_Z = 0.05

wrist_filter_x = None
wrist_filter_y = None
wrist_filter_z = None


def main():
    model_path = files("dual_arms.teleop.models").joinpath("hand_landmarker.task")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    VER_SCALE = 1.0     # Reference for Wrist -> Middle Knuckle
    HOR_SCALE = 0.65    # Reference for Index -> Pinky (Palm Width)

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame_bgr = cap.read()
            if not ret: break

            frame_bgr = cv2.flip(frame_bgr, 1)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = frame_bgr.shape

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                hand_norm = result.hand_landmarks[0]
                
                # --- 1. X/Y Tracking ---
                wrist = hand_norm[0]
                raw_x, raw_y = wrist.x, wrist.y

                # --- 2. Robust Z Tracking (Dual Axis) ---
                
                # Points for Vertical Axis (Wrist to Middle Knuckle)
                p0 = hand_norm[0]
                p9 = hand_norm[9]
                dist_vert = math.sqrt(((p0.x-p9.x)*w)**2 + ((p0.y-p9.y)*h)**2)

                # Points for Horizontal Axis (Index to Pinky)
                p5 = hand_norm[5]
                p17 = hand_norm[17]
                dist_horiz = math.sqrt(((p5.x-p17.x)*w)**2 + ((p5.y-p17.y)*h)**2)

                # Calculate two potential Depth values
                # We normalize by the physical size constants
                # Multiplier 1000 is for readability
                if dist_vert > 0:
                    z_vert = (1000 * VER_SCALE) / dist_vert
                else: 
                    z_vert = 9999

                if dist_horiz > 0:
                    z_horiz = (1000 * HOR_SCALE) / dist_horiz
                else:
                    z_horiz = 9999

                # THE TRICK: Take the MINIMUM Z (Closeset)
                # Foreshortening makes objects look smaller (farther).
                # The axis facing the camera most directly will yield the smallest Z value.
                raw_z = min(z_vert, z_horiz)

                # --- 3. Filtering ---
                global wrist_filter_x, wrist_filter_y, wrist_filter_z
                if wrist_filter_x is None:
                    wrist_filter_x = OneEuroFilter(freq=FPS, mincutoff=MIN_CUTOFF, beta=BETA_XY)
                    wrist_filter_y = OneEuroFilter(freq=FPS, mincutoff=MIN_CUTOFF, beta=BETA_XY)
                    wrist_filter_z = OneEuroFilter(freq=FPS, mincutoff=MIN_CUTOFF_Z, beta=BETA_Z)

                t = time.time()
                filt_x = wrist_filter_x(raw_x, timestamp=t)
                filt_y = wrist_filter_y(raw_y, timestamp=t)
                filt_z = wrist_filter_z(raw_z, timestamp=t)

                norm_z = (filt_z - 9) / (17 - 9)
                
                # Clamp the value so it stays strictly between 0 and 1
                norm_z = max(0.0, min(1.0, norm_z))

                # --- 4. Visualization ---
                cx, cy = int(filt_x * w), int(filt_y * h)
                cv2.circle(frame_bgr, (cx, cy), 10, (0, 255, 0), -1)

                # Draw the axis being used currently
                # If Vertical Z is the chosen one, draw yellow line on vertical bone
                if z_vert < z_horiz:
                    start = (int(p0.x*w), int(p0.y*h))
                    end = (int(p9.x*w), int(p9.y*h))
                    cv2.line(frame_bgr, start, end, (0, 255, 255), 3) # Yellow = Vertical
                else:
                    start = (int(p5.x*w), int(p5.y*h))
                    end = (int(p17.x*w), int(p17.y*h))
                    cv2.line(frame_bgr, start, end, (255, 0, 255), 3) # Magenta = Horizontal

                center_x = (filt_x - 0.5) * 2.0

                # Step 2: (Optional) Add a "Deadzone"
                # This makes sure the robot stops completely when your hand is near the center
                DEADZONE = 0.1
                if abs(center_x) < DEADZONE:
                    center_x = 0.0
                
                # Step 3: Clamp to ensure it never exceeds -1 or 1
                center_x = max(0, min(1.0, center_x))

                cv2.putText(frame_bgr, f"x: {center_x:.2f} y: {filt_y:.2f} Z: {norm_z:.2f}", (cx + 15, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            cv2.imshow("MediaPipe Hands", frame_bgr)
            if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()