# Gesture based sign detection - palm, thumbs up , okay , peace and fist

import mediapipe as mp
import cv2 as cv
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv.VideoCapture(0)
width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)


#detecting left and right hands
def get_label(index,hand,results):
    output = None
    wrist = hand.landmark[0]
    thumb_tip = hand.landmark[4]
    index_mcp = hand.landmark[5]

    for idx , classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index :
            label = classification.classification[0].label
            score = classification.classification[0].score
            if thumb_tip.x < index_mcp.x and thumb_tip.x < wrist.x :
                label = "Right"
            else :
                label = "Left"
            text = '{} {}'.format(label, round(score,2))

            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                (width, height)).astype(int)
                           )
            output = text, coords
    return output






#detecting angles between 3 landmarks
def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = b - a
    bc = b - c

    cosine_angle = np.dot(ba,bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

joint_list = [[4,3,2],[8,7,6],[12,11,10],[16,15,14],[20,19,18]]

#getting label for every finger
def finger_label(results,img):

    finger_states = []
    symbol = ''

    if results.multi_hand_landmarks:

        for hand in results.multi_hand_landmarks:

            for joint in joint_list:

                a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y])
                b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y])
                c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y])

                angle = calculate_angle(a, b, c)

                if angle > 165 :
                    finger_state = 1
                else:
                    finger_state = 0
                finger_states.append(finger_state)
            if finger_states == [1,0,0,0,0] :
                symbol = 'Thumbs up'
            if finger_states == [0,1,1,0,0]:
                symbol = 'Peace'
            if finger_states == [0,0,0,0,0]:
                symbol = 'Fist'
            if finger_states == [1,1,1,1,1]:
                symbol = 'Palm'
            if finger_states == [0,0,1,1,1]:
                symbol = 'Okay'
            wrist = hand.landmark[0]
            x = int(wrist.x*width)
            y = int(wrist.y*height)
            cv.putText(img, str(symbol), (x,y+50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv.putText(
        img,
        str(finger_states),
        (20,50),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )
    return img



#drawing the angles on the screen
def draw_angles(image,joint_list,results):
    for hand in results.multi_hand_landmarks :
        for joint in joint_list :
            a = np.array([hand.landmark[joint[0]].x,hand.landmark[joint[0]].y])
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y])
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y])
            print(hand.landmark[joint[0]].x , hand.landmark[joint[0]])
            angle = calculate_angle(a,b,c)
            if angle > 180 :
                angle = 360 - angle


            cv.putText(image,str(round(angle,2)),tuple(np.multiply(b,(width,height)).astype(int)),cv.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)


    return image



with mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5) as hands :
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret :
            print('unable to access the camera')
            break

        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img = cv.flip(img,1)
        img.flags.writeable = False
        results = hands.process(img)
        img.flags.writeable = True
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        if results.multi_hand_landmarks :
            for num , hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(img,hand,mp_hands.HAND_CONNECTIONS)

                label_info = get_label(num, hand, results)
                if label_info:
                    text, coords = label_info
                    cv.putText(img, text, coords, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            draw_angles(img,joint_list,results)
            finger_label(results,img)






        cv.imshow('webcam footage',img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            print('closing camera')
            break


cap.release()
cv.destroyAllWindows()

