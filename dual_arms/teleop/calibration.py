import time
import cv2
import math
import mediapipe as mp
from importlib.resources import files

# --- CONFIGURATION ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def main():
    model_path = files("dual_arms.teleop.models").joinpath("hand_landmarker.task")
    
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1, # Calibrate one hand at a time
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)

    # We fix Vertical Scale to 1.0 as the reference.
    # We will find the Horizontal Scale relative to this.
    VER_SCALE = 1.0 

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
                
                # --- 1. Get Pixel Distances ---
                
                # Vertical Axis (Wrist 0 -> Middle 9)
                p0, p9 = hand_norm[0], hand_norm[9]
                dist_vert = math.sqrt(((p0.x-p9.x)*w)**2 + ((p0.y-p9.y)*h)**2)

                # Horizontal Axis (Index 5 -> Pinky 17)
                p5, p17 = hand_norm[5], hand_norm[17]
                dist_horiz = math.sqrt(((p5.x-p17.x)*w)**2 + ((p5.y-p17.y)*h)**2)

                # --- 2. Calculate the Ratio ---
                # We want z_vert == z_horiz.
                # Logic: (1000 * VER) / dist_vert == (1000 * HOR) / dist_horiz
                # Therefore: HOR = VER * (dist_horiz / dist_vert)
                
                if dist_vert > 0:
                    suggested_hor_scale = VER_SCALE * (dist_horiz / dist_vert)
                else:
                    suggested_hor_scale = 0

                # --- 3. Visualization ---
                
                # Draw Box for clarity
                cv2.rectangle(frame_bgr, (10, 10), (450, 160), (0, 0, 0), -1)
                
                # Instructions
                cv2.putText(frame_bgr, "HOLD HAND FLAT FACING CAMERA", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Show Live Data
                # If this number is stable, THAT is your HOR_SCALE value.
                cv2.putText(frame_bgr, f"Suggested HOR_SCALE: {suggested_hor_scale:.3f}", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                cv2.putText(frame_bgr, f"Dist Vert (px): {int(dist_vert)}", (20, 115), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(frame_bgr, f"Dist Horiz (px): {int(dist_horiz)}", (20, 140), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                # Visual Lines
                start_v = (int(p0.x*w), int(p0.y*h))
                end_v = (int(p9.x*w), int(p9.y*h))
                cv2.line(frame_bgr, start_v, end_v, (0, 255, 255), 2) # Yellow Vertical

                start_h = (int(p5.x*w), int(p5.y*h))
                end_h = (int(p17.x*w), int(p17.y*h))
                cv2.line(frame_bgr, start_h, end_h, (255, 0, 255), 2) # Magenta Horizontal

            cv2.imshow("Calibration Mode", frame_bgr)
            if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()