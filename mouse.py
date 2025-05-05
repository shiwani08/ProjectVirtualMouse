import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import threading
import screen_brightness_control as sbc
import time
import datetime  # Make sure this is at the top of your file



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

# State flags and variables
brightness_mode = False
brightness_anchor_y = None
last_brightness_time = 0
brightness = sbc.get_brightness(display=0)[0]
last_screenshot_time = 0

zoom_mode = False
zoom_anchor_y = None
last_zoom_time = 0
zoom_level = 100
zoom_prev_level = 100  # Initialize zoom previous level
last_zoom_keypress_time = 0  # Add timer for keypress throttling

# Alt+Tab Mode
alt_tab_mode = False
alt_tab_time = 0
last_tab_time = 0

while True:
    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None

    if frame is None:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(frame_rgb)

    h, w, _ = frame.shape

    if results.multi_hand_landmarks and results.multi_handedness:
        hands = list(zip(results.multi_hand_landmarks, results.multi_handedness))
        right_hand = None
        left_hand = None

        for hand_landmarks, handedness in hands:
            label = handedness.classification[0].label
            if label == 'Right':
                right_hand = hand_landmarks
            elif label == 'Left':
                left_hand = hand_landmarks

        # Alt+Tab using right hand fist gesture (excluding thumb)
        if right_hand:
            rh = right_hand.landmark
            thumb_tip = rh[4]
            thumb_ip = rh[3]  # Thumb joint
            index_tip = rh[8]
            middle_tip = rh[12]
            ring_tip = rh[16]
            pinky_tip = rh[20]

            # Detect: thumb extended AND other fingers bent
            thumb_extended = thumb_tip.x < rh[2].x and thumb_tip.y < rh[3].y  # leftward & upward thumb
            index_bent = index_tip.y > rh[6].y
            middle_bent = middle_tip.y > rh[10].y
            ring_bent = ring_tip.y > rh[14].y
            pinky_bent = pinky_tip.y > rh[18].y

            if thumb_extended and index_bent and middle_bent and ring_bent and pinky_bent:
                if not alt_tab_mode:
                    alt_tab_mode = True
                    pyautogui.keyDown('alt')
                    alt_tab_time = time.time()
                    cv2.putText(frame, "ALT+TAB MODE ACTIVE", (int(w / 2) - 150, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Simulate "Tab" key press using left hand (thumb and index close together)
        if left_hand and alt_tab_mode:
            lh_landmarks = left_hand.landmark
            thumb_tip = lh_landmarks[4]
            index_tip = lh_landmarks[8]

            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

            # Check if thumb and index are close together
            if abs(thumb_x - index_x) < 40 and abs(thumb_y - index_y) < 40:
                if time.time() - last_tab_time > 0.5:  # Add delay to simulate Tab key press
                    pyautogui.press('tab')  # Simulate the Tab key press
                    last_tab_time = time.time()
                    cv2.putText(frame, "Tab Pressed", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Check to deactivate ALT+TAB mode when right hand is no longer in fist
        if alt_tab_mode and right_hand:
            rh_landmarks = right_hand.landmark
            thumb_tip = rh_landmarks[4]
            index_tip = rh_landmarks[8]
            middle_tip = rh_landmarks[12]
            ring_tip = rh_landmarks[16]
            pinky_tip = rh_landmarks[20]

            thumb_extended = thumb_tip.y < rh_landmarks[3].y
            other_fingers_fisted = (index_tip.y > rh_landmarks[6].y and
                                    middle_tip.y > rh_landmarks[10].y and
                                    ring_tip.y > rh_landmarks[14].y and
                                    pinky_tip.y > rh_landmarks[18].y)

            if not (thumb_extended and other_fingers_fisted):
                if alt_tab_mode:
                    alt_tab_mode = False
                    pyautogui.keyUp('alt')  # Release the Alt key
                    cv2.putText(frame, "ALT+TAB MODE DEACTIVATED", (int(w / 2) - 150, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if results.multi_hand_landmarks and results.multi_handedness:
        hands = list(zip(results.multi_hand_landmarks, results.multi_handedness))
        right_hand = None
        left_hand = None

        for hand_landmarks, handedness in hands:
            label = handedness.classification[0].label
            if label == 'Right':
                right_hand = hand_landmarks
            elif label == 'Left':
                left_hand = hand_landmarks
        # Screenshot with "L" gesture (refined)
        if left_hand:
            landmarks = left_hand.landmark
            thumb_tip = landmarks[4]
            thumb_mcp = landmarks[2]
            index_tip = landmarks[8]
            index_mcp = landmarks[5]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]

            # Thumb pointing right
            thumb_horizontal = thumb_tip.x - thumb_mcp.x > 0.1
            # Index pointing up
            index_vertical = index_mcp.y - index_tip.y > 0.1
            # Other fingers folded
            middle_folded = middle_tip.y > landmarks[10].y
            ring_folded = ring_tip.y > landmarks[14].y
            pinky_folded = pinky_tip.y > landmarks[18].y

            if thumb_horizontal and index_vertical and middle_folded and ring_folded and pinky_folded:
                if time.time() - last_screenshot_time > 2:
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    pyautogui.screenshot(f'screenshot_{timestamp}.png')
                    last_screenshot_time = time.time()
                    cv2.putText(frame, "Screenshot Taken", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Fist swipe to switch desktops (requires both hands, blocks other functions)
        if right_hand is not None and left_hand is not None and not brightness_mode and not zoom_mode:
            rh = right_hand.landmark
            lh = left_hand.landmark

            # Detect if both hands are forming fists
            def is_fist(landmarks):
                return (landmarks[8].y > landmarks[6].y and
                        landmarks[12].y > landmarks[10].y and
                        landmarks[16].y > landmarks[14].y and
                        landmarks[20].y > landmarks[18].y)

            right_fist = is_fist(rh)
            left_fist = is_fist(lh)

            if right_fist and left_fist:
                # Track movement over time
                if 'swipe_anchor_r' not in globals():
                    swipe_anchor_r = rh[0].x
                    swipe_anchor_l = lh[0].x
                    swipe_start_time = time.time()
                    swipe_detected = False
                else:
                    if not swipe_detected:
                        move_r = rh[0].x - swipe_anchor_r
                        move_l = lh[0].x - swipe_anchor_l

                        if abs(move_r) > 0.15 and abs(move_l) > 0.15:
                            if move_r < 0 and move_l < 0:
                                pyautogui.hotkey('ctrl', 'winleft', 'left')
                                cv2.putText(frame, "Desktop Left", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                                swipe_detected = True
                            elif move_r > 0 and move_l > 0:
                                pyautogui.hotkey('ctrl', 'winleft', 'right')
                                cv2.putText(frame, "Desktop Right", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                                swipe_detected = True

                # Reset after timeout
                if time.time() - swipe_start_time > 2:
                    del swipe_anchor_r, swipe_anchor_l, swipe_start_time, swipe_detected
        
        # Zoom control
        if right_hand:
            rh_landmarks = right_hand.landmark
            thumb_tip = rh_landmarks[4]
            middle_tip = rh_landmarks[12]

            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            middle_x, middle_y = int(middle_tip.x * w), int(middle_tip.y * h)

            dist_zoom = np.hypot(thumb_x - middle_x, thumb_y - middle_y)

            if dist_zoom < 40 and not zoom_mode:
                zoom_mode = True
                zoom_anchor_y = middle_y
                last_zoom_time = time.time()
                # Show zoom mode activated message
                cv2.putText(frame, "ZOOM MODE ACTIVATED", (int(w/2)-150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if zoom_mode:
                # Show zoom mode is active
                cv2.putText(frame, "ZOOM MODE ACTIVE", (int(w/2)-100, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                           
                dy = zoom_anchor_y - middle_y
                delta_zoom = int(dy / 3)
                new_zoom = np.clip(zoom_level + delta_zoom, 50, 150)

                # Only send zoom keypress if zoom level changes AND enough time has passed
                current_time = time.time()
                if new_zoom != zoom_prev_level and current_time - last_zoom_keypress_time > 0.3:  # 300ms delay between keypresses
                    if new_zoom > zoom_prev_level:
                        pyautogui.hotkey('ctrl', '+')
                        # Debug visualization
                        cv2.putText(frame, "ZOOM IN", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        # Release any keys that might be held down
                       
                        # Small delay
                        time.sleep(0.05)
                        # Press and release zoom in keys
                        
                    elif new_zoom < zoom_prev_level:
                        pyautogui.hotkey('ctrl', '-')
                        # Debug visualization
                        cv2.putText(frame, "ZOOM OUT", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        # Release any keys that might be held down
                        # Small delay
                        time.sleep(0.05)
                        # Press and release zoom out keys
                    
                    zoom_prev_level = new_zoom
                    last_zoom_keypress_time = current_time
                
                # Draw zoom bar
                bar_x, bar_y = w - 80, 100
                bar_height = 300
                fill_height = int((new_zoom - 50) / 100 * bar_height)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 30, bar_y + bar_height), (50, 50, 50), 2)
                cv2.rectangle(frame, (bar_x, bar_y + bar_height - fill_height),
                              (bar_x + 30, bar_y + bar_height), (0, 200, 200), -1)
                cv2.putText(frame, f'{new_zoom}%', (bar_x - 20, bar_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if dist_zoom > 60 and time.time() - last_zoom_time > 1:
                    zoom_level = new_zoom
                    zoom_mode = False
                    zoom_anchor_y = None
                    # Show zoom mode deactivated message
                    cv2.putText(frame, "ZOOM MODE DEACTIVATED", (int(w/2)-150, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Brightness control
        if right_hand and not zoom_mode:
            rh_landmarks = right_hand.landmark
            thumb_tip = rh_landmarks[4]
            index_tip = rh_landmarks[8]

            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

            dist_brightness = np.hypot(thumb_x - index_x, thumb_y - index_y)

            if dist_brightness < 40 and not brightness_mode:
                brightness_mode = True
                brightness_anchor_y = index_y
                last_brightness_time = time.time()

            if brightness_mode:
                dy = brightness_anchor_y - index_y
                delta = int(dy / 5)
                new_brightness = np.clip(brightness + delta, 0, 100)
                sbc.set_brightness(new_brightness)

                # Draw brightness bar
                bar_x, bar_y = 50, 100
                bar_height = 300
                fill_height = int((new_brightness / 100) * bar_height)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 30, bar_y + bar_height), (50, 50, 50), 2)
                cv2.rectangle(frame, (bar_x, bar_y + bar_height - fill_height),
                              (bar_x + 30, bar_y + bar_height), (0, 255, 255), -1)
                cv2.putText(frame, f'{new_brightness}%', (bar_x, bar_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if dist_brightness > 60 and time.time() - last_brightness_time > 1:
                    brightness = new_brightness
                    brightness_mode = False
                    brightness_anchor_y = None

        # Cursor and clicks (disabled during brightness/zoom)
        if right_hand and not brightness_mode and not zoom_mode:
            index_tip = right_hand.landmark[8]
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

        if left_hand and not brightness_mode and not zoom_mode:
            landmarks = left_hand.landmark
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]

            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            middle_x, middle_y = int(middle_tip.x * w), int(middle_tip.y * h)

            cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 255, 0), -1)
            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)
            cv2.circle(frame, (middle_x, middle_y), 10, (0, 0, 255), -1)

            if abs(thumb_x - index_x) < 40 and abs(thumb_y - index_y) < 40:
                pyautogui.rightClick()
                cv2.putText(frame, "Right Click", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                pyautogui.sleep(1)
            elif abs(thumb_x - middle_x) < 40 and abs(thumb_y - middle_y) < 40:
                pyautogui.click()
                cv2.putText(frame, "Left Click", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                pyautogui.sleep(1)

        # Draw landmarks

        for hand_landmarks, _ in hands:
            drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    cv2.imshow("Virtual Mouse - Brightness & Zoom", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()