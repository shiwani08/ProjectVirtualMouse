import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
index_y = 0

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    print(hands)

    if hands:
        for hand in hands:
            drawing.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:  # Tip of the index finger
                    cv2.circle(img = frame, center = (x, y), radius = 15, color = (0, 0, 255))
                    index_x = screen_width / frame_width * x
                    index_y = screen_height / frame_height * y
                    pyautogui.moveTo(index_x, index_y)

                if id == 4:  # Tip of the thumb
                    cv2.circle(img = frame, center = (x, y), radius = 15, color = (255, 0, 0))
                    thumb_x = screen_width / frame_width * x
                    thumb_y = screen_height / frame_height * y
                    print('outside', abs(index_y - thumb_y))

                    # Left Click when index and thumb are close
                    if abs(index_y - thumb_y) < 60:
                        pyautogui.click()
                        pyautogui.sleep(1)

                # Right Click when thumb and middle finger are close
                if id == 12:  # Tip of the middle finger
                    middle_x = int(landmarks[12].x * frame_width)
                    middle_y = int(landmarks[12].y * frame_height)
                    print('middle finger', abs(thumb_y - middle_y))

                    # Right Click when thumb and middle finger are close
                    if abs(thumb_y - middle_y) < 60:
                        pyautogui.rightClick()
                        pyautogui.sleep(1)

    cv2.imshow("This is your HANDY mouse", frame)
    cv2.waitKey(1)
