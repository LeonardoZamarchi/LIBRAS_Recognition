import cv2
import mediapipe as mp
import os
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

IMAGE_FILES = []
for i in os.listdir():
    if i.find('.jpg')>0:
        IMAGE_FILES.append(i)
#IMAGE_FILES = ['tst.jpg']

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.flip(cv2.imread(file), 1)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
        print('####################################################################################')
        print('id = 0,' + str({hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width})  +',' + str({hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height}))
        print('id = 1,' + str({hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width})  +',' + str({hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height}))
        print('id = 2,' + str({hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width})  +',' + str({hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height}))
        print('id = 3,' + str({hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width})   +',' + str({hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height}))
        print('id = 4,' + str({hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width})  +',' + str({hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height}))
        print('id = 5,' + str({hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width})   +',' + str({hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height}))
        print('id = 6,' + str({hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width})   +',' + str({hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height}))
        print('id = 7,' + str({hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width})   +',' + str({hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height}))
        print('id = 8,' + str({hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width})   +',' + str({hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height}))
        print('id = 9,' + str({hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width})  +',' + str({hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height}))
        print('id = 10,' + str({hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width}) +','  + str({hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height}))
        print('id = 11,' + str({hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width}) +','  + str({hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height}))
        print('id = 12,' + str({hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width}) +','  + str({hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height}))
        print('id = 13,' + str({hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width})   +','  + str({hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height}))
        print('id = 14,' + str({hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width})   +','  + str({hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height}))
        print('id = 15,' + str({hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width})   +','  + str({hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height}))
        print('id = 16,' + str({hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width})   +','  + str({hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height}))
        print('id = 17,' + str({hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width}) +','  + str({hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height}))
        print('id = 18,' + str({hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width}) +','  + str({hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height}))
        print('id = 19,' + str({hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width}) +','  + str({hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height}))
        print('id = 20,' + str({hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width}) +','  + str({hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height}))

        mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
