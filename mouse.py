import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

# Initialize previous coordinates for cursor smoothing
prev_x, prev_y = 0, 0
smooth_factor = 0.2  # Adjust this for more/less smoothing

# Initialize previous y-coordinate for scroll detection
prev_index_y = 0

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            # Initialize variables to hold finger positions
            index_x, index_y = 0, 0
            thumb_x, thumb_y = 0, 0
            middle_x, middle_y = 0, 0

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:  # Tip of the index finger (cursor movement)
                    index_x = screen_width / frame_width * x
                    index_y = screen_height / frame_height * y
                    cv2.circle(img=frame, center=(x, y), radius=15, color=(0, 0, 255))

                if id == 4:  # Tip of the thumb
                    thumb_x = screen_width / frame_width * x
                    thumb_y = screen_height / frame_height * y
                    cv2.circle(img=frame, center=(x, y), radius=15, color=(255, 0, 0))

                if id == 12:  # Tip of the middle finger
                    middle_x = screen_width / frame_width * x
                    middle_y = screen_height / frame_height * y
                    cv2.circle(img=frame, center=(x, y), radius=15, color=(0, 255, 0))

            # Smooth cursor movement using weighted average
            if index_x != 0 and index_y != 0:
                smooth_x = prev_x + (index_x - prev_x) * smooth_factor
                smooth_y = prev_y + (index_y - prev_y) * smooth_factor
                pyautogui.moveTo(smooth_x, smooth_y)
                prev_x, prev_y = smooth_x, smooth_y

            # Right-click when index and thumb are close
            if abs(index_y - thumb_y) < 50 and abs(index_x - thumb_x) < 50:
                pyautogui.rightClick()
                pyautogui.sleep(1)
                print("right Click")

            # Left-click when thumb and middle finger are close
            elif abs(thumb_y - middle_y) < 50 and abs(thumb_x - middle_x) < 50:
                pyautogui.click()
                pyautogui.sleep(1)
                print("Left Click")

            # Scroll when index and middle fingers are close
            if abs(index_y - middle_y) < 50:
                if index_y < prev_index_y:  # Scroll up
                    pyautogui.scroll(20)
                    print("Scrolling Up")
                elif index_y > prev_index_y:  # Scroll down
                    pyautogui.scroll(-20)
                    print("Scrolling Down")
                prev_index_y = index_y

    cv2.imshow("This is your HANDY mouse", frame)
    cv2.waitKey(1)
