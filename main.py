import mediapipe as mp
import cv2
import pyautogui
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

upShiftCoolDown = time.perf_counter()
downShiftCoolDown = time.perf_counter()
avgX = 0


def thumbDown(hand):
  if (hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y >
          hand.landmark[mp_hands.HandLandmark.THUMB_IP].y):
    return True
  return False

def indexDown(hand):
  if (hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y >
          hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y):
      return True
  return False

def middleDown(hand):
  if (hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y >
          hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y):
    return True
  return False

def ringDown(hand):
  if (hand.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y >
          hand.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y):
    return True
  return False

def pinkyDown(hand):
  if (hand.landmark[mp_hands.HandLandmark.PINKY_TIP].y >
          hand.landmark[mp_hands.HandLandmark.PINKY_DIP].y):
    return True
  return False

def handedness(hand):  # example param: results.multi_hand_landmarks[1]
  if (hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x <
          hand.landmark[mp_hands.HandLandmark.PINKY_TIP].x):
    return("Right")
  else:
    return("Left")

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
      #for hand_landmarks in results.multi_hand_landmarks:
        #mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

      for hand_landmarks in results.multi_hand_landmarks:
        if (handedness(hand_landmarks) == "Right"):

          if (indexDown(hand_landmarks)):
            pyautogui.keyDown('w')
          else:
            pyautogui.keyUp('w')

          if (thumbDown(hand_landmarks) and (time.perf_counter()-upShiftCoolDown) > 0.5):
            pyautogui.press('e')
            upShiftCoolDown = time.perf_counter()


        elif (handedness(hand_landmarks) == "Left"):

          if (indexDown(hand_landmarks)):
            pyautogui.keyDown('s')
          else:
            pyautogui.keyUp('s')

          if (thumbDown((hand_landmarks)) and (time.perf_counter()-downShiftCoolDown) > 0.5):
            pyautogui.press('q')
            downShiftCoolDown = time.perf_counter()

        avgX += hand_landmarks.landmark[0].x
      avgX /= len(results.multi_hand_landmarks)
      if avgX < 0.42:
        pyautogui.keyDown('a')
      elif avgX > 0.58:
        pyautogui.keyDown('d')
      else:
        pyautogui.keyUp('a')
        pyautogui.keyUp('d')
      avgX = 0


    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
