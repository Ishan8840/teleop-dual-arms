import time
import math
import cv2
import mediapipe as mp
from importlib.resources import files
import numpy as np

# ---- MediaPipe shortcuts ----
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def main():
    # Load your existing hand_landmarker model
    model_path = str(files("dual_arms.teleop.models").joinpath("hand_landmarker.task"))

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarker = HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    print("Depth calibration started.")
    print("Instructions:")
    print("  - Put your hand in front of the camera, fingers spread.")
    print("  - Keep a similar pose to what you’ll use during teleop.")
    print("  - Press 'c' to capture a sample (do this at a few depths).")
    print("  - Press 'q' to finish and compute scale constants.\n")

    vert_over_horiz = []  # dist_vert / dist_horiz
    horiz_over_vert = []  # dist_horiz / dist_vert

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read from camera.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        timestamp_ms = int(time.time() * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        current_vert = None
        current_horiz = None

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            # Landmarks:
            # 0: wrist, 9: middle finger base
            # 5: index finger base, 17: pinky base
            p0, p9 = hand[0], hand[9]
            p5, p17 = hand[5], hand[17]

            # Distances in pixels
            dist_vert = math.sqrt(((p0.x - p9.x) * w) ** 2 + ((p0.y - p9.y) * h) ** 2)
            dist_horiz = math.sqrt(((p5.x - p17.x) * w) ** 2 + ((p5.y - p17.y) * h) ** 2)

            current_vert = dist_vert
            current_horiz = dist_horiz

            # Draw the landmarks used for visual feedback
            cx0, cy0 = int(p0.x * w), int(p0.y * h)
            cx9, cy9 = int(p9.x * w), int(p9.y * h)
            cx5, cy5 = int(p5.x * w), int(p5.y * h)
            cx17, cy17 = int(p17.x * w), int(p17.y * h)

            cv2.circle(frame, (cx0, cy0), 6, (0, 255, 0), -1)
            cv2.circle(frame, (cx9, cy9), 6, (0, 255, 0), -1)
            cv2.line(frame, (cx0, cy0), (cx9, cy9), (0, 255, 0), 2)

            cv2.circle(frame, (cx5, cy5), 6, (255, 0, 0), -1)
            cv2.circle(frame, (cx17, cy17), 6, (255, 0, 0), -1)
            cv2.line(frame, (cx5, cy5), (cx17, cy17), (255, 0, 0), 2)

            cv2.putText(
                frame,
                f"vert: {dist_vert:.1f}px horiz: {dist_horiz:.1f}px",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

        cv2.putText(
            frame,
            "Press 'c' to capture, 'q' to quit",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

        cv2.imshow("Depth Calibration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("c") and current_vert is not None and current_horiz is not None:
            if current_horiz > 0 and current_vert > 0:
                vh = current_vert / current_horiz
                hv = current_horiz / current_vert
                vert_over_horiz.append(vh)
                horiz_over_vert.append(hv)
                print(f"Captured sample: vert={current_vert:.1f}, horiz={current_horiz:.1f}, vert/horiz={vh:.3f}")
            else:
                print("Skipped sample (zero distance).")

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()

    if not vert_over_horiz:
        print("No samples captured. Run again and press 'c' with your hand visible.")
        return

    # Compute averages
    mean_vh = float(np.mean(vert_over_horiz))
    mean_hv = float(np.mean(horiz_over_vert))

    print("\n--- Calibration Results ---")
    print(f"Number of samples: {len(vert_over_horiz)}")
    print(f"Average dist_vert / dist_horiz: {mean_vh:.4f}")
    print(f"Average dist_horiz / dist_vert: {mean_hv:.4f}")

    # Suppose we keep K_VERT = 1000.0 (your current vertical constant)
    K_VERT_BASE = 1000.0

    # We want: K_VERT / dist_vert ≈ K_HORIZ / dist_horiz
    # => K_VERT / K_HORIZ ≈ dist_vert / dist_horiz
    # => K_HORIZ ≈ K_VERT / (dist_vert / dist_horiz)
    # Use the average ratio:
    K_HORIZ_CALIB = K_VERT_BASE / mean_vh

    print("\nSuggested scale constants (to replace 1000.0 and 650.0):")
    print(f"  K_VERT  (for p0-p9)      = {K_VERT_BASE:.3f}")
    print(f"  K_HORIZ (for p5-p17)     = {K_HORIZ_CALIB:.3f}")

    print("\nSo in your HandTracker.get_target() you can do:")
    print("  z_vert  = (1000.0 / dist_vert)")
    print(f"  z_horiz = ({K_HORIZ_CALIB:.3f} / dist_horiz)")
    print("\nThis makes the two depth metrics better aligned in scale.")


if __name__ == '__main__':
    main()
