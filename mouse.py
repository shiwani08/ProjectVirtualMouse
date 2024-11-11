import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

# Initialize previous finger y-coordinates for scroll detection
index_y = 0
middle_y = 0
thumb_y = 0

# Add variables for previous distances (for better scroll accuracy)
prev_index_y = 0
prev_middle_y = 0

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
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:  # Tip of the index finger
                    cv2.circle(img=frame, center=(x, y), radius=15, color=(0, 0, 255))
                    index_x = screen_width / frame_width * x
                    index_y = screen_height / frame_height * y
                    pyautogui.moveTo(index_x, index_y)

                if id == 4:  # Tip of the thumb
                    cv2.circle(img=frame, center=(x, y), radius=15, color=(255, 0, 0))
                    thumb_x = screen_width / frame_width * x
                    thumb_y = screen_height / frame_height * y
                    # Left Click when index and thumb are close
                    if abs(index_y - thumb_y) < 60:
                        pyautogui.click()
                        pyautogui.sleep(1)

                if id == 12:  # Tip of the middle finger
                    middle_x = int(landmarks[12].x * frame_width)
                    middle_y = int(landmarks[12].y * frame_height)

                    # Right Click when thumb and middle finger are close
                    if abs(thumb_y - middle_y) < 60:
                        pyautogui.rightClick()
                        pyautogui.sleep(1)

            # Scroll functionality based on distance between index and middle finger
            # Scroll condition based on the change in the distance between index and middle fingers
            if abs(index_y - middle_y) < 60:  # When fingers are close enough to scroll
                if index_y < middle_y:  # Scroll Up
                    pyautogui.scroll(10)
                    print("Scrolling Up")
                elif middle_y < index_y:  # Scroll Down
                    pyautogui.scroll(-10)
                    print("Scrolling Down")

                # Prevent unnecessary scrolling by comparing the previous values for smoother control
                if abs(prev_index_y - index_y) > 10 or abs(prev_middle_y - middle_y) > 10:
                    prev_index_y = index_y
                    prev_middle_y = middle_y

    cv2.imshow("This is your HANDY mouse", frame)
    cv2.waitKey(1)
