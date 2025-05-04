import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import threading
import screen_brightness_control as sbc

# Globals for thread-safe frame sharing
latest_frame = None
frame_lock = threading.Lock()

# Webcam capture in background thread
def capture_frames():
    global latest_frame
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            continue
        frame = cv2.flip(frame, 1)
        with frame_lock:
            latest_frame = frame.copy()

# Start frame capture thread
threading.Thread(target=capture_frames, daemon=True).start()

# Setup MediaPipe + screen settings
hands_detector = mp.solutions.hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.7
)
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
circle_x, circle_y = 0, 0
smoothing_delay = 0.33

while True:
    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None

    if frame is None:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(frame_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            if hand_handedness.classification[0].label == 'Right':
                h, w, _ = frame.shape
                index_tip = hand_landmarks.landmark[8]

                raw_x = int(index_tip.x * w)
                raw_y = int(index_tip.y * h)
                cv2.circle(frame, (raw_x, raw_y), 8, (0, 255, 255), -1)

                target_x = np.interp(index_tip.x, [0.1, 0.90], [0, screen_width])
                target_y = np.interp(index_tip.y, [0.1, 0.90], [0, screen_height])

                circle_x += (target_x - circle_x) * smoothing_delay
                circle_y += (target_y - circle_y) * smoothing_delay

                pyautogui.moveTo(circle_x, circle_y)

                screen_dot_x = int(np.interp(index_tip.x, [0.1, 0.9], [0, w]))
                screen_dot_y = int(np.interp(index_tip.y, [0.1, 0.9], [0, h]))
                cv2.circle(frame, (screen_dot_x, screen_dot_y), 20, (255, 0, 255), 2)
            
            elif hand_handedness.classification[0].label == 'Left':
                landmarks = hand_landmarks.landmark
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                middle_tip = landmarks[12]

                # Convert to pixel coordinates
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
                middle_x, middle_y = int(middle_tip.x * w), int(middle_tip.y * h)

                # Visual debug (optional)
                cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 255, 0), -1)
                cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)
                cv2.circle(frame, (middle_x, middle_y), 10, (0, 0, 255), -1)

                # Right Click: Thumb & Index close
                if abs(thumb_x - index_x) < 40 and abs(thumb_y - index_y) < 40:
                    pyautogui.rightClick()
                    cv2.putText(frame, "left Click", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    pyautogui.sleep(1)

                # Left Click: Thumb & Middle close
                elif abs(thumb_x - middle_x) < 40 and abs(thumb_y - middle_y) < 40:
                    pyautogui.click()
                    cv2.putText(frame, "Right Click", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    pyautogui.sleep(1)

            drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            

    cv2.imshow("Smoothed Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
