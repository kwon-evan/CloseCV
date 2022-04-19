import math
import sys
import dlib
import cv2
import numpy as np


def rotate_image(image, landmarks):
    tan_theta = (shape_2d[30][0] - shape_2d[27][0]) / (shape_2d[30][1] - shape_2d[27][1])
    theta = np.arctan(tan_theta)
    angle = -1 * theta * 180 / math.pi

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return result


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
cam = cv2.VideoCapture(0)

color_blue = (255, 0, 0)
color_green = (0, 255, 0)
color_red = (0, 0, 255)
depth = 1000

w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

while True:
    ret_val, img = cam.read()
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    origin = img.copy()
    rotated_ROI = ROI = origin[0:147, 0:236]

    faces = detector(rgb_img)
    for face in faces:
        dlib_shape = predictor(rgb_img, face)
        shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

        center_X, center_Y = np.mean(shape_2d, axis=0).astype(np.int)
        center = np.mean(shape_2d, axis=0).astype(np.int)

        roi_h = face.bottom() - face.top()
        roi_w = w / h * roi_h

        # draw line, cam center to roi center
        cv2.line(img,
                 (int(w / 2), int(h / 2)),
                 (center_X, center_Y),
                 color=color_blue,
                 thickness=2)
        cv2.circle(img,
                   center=(int(w / 2), int(h / 2)),
                   radius=1,
                   color=color_green,
                   thickness=2)
        cv2.circle(img,
                   center=(center_X, center_Y),
                   radius=1,
                   color=color_red,
                   thickness=2)

        print(f'x오차 : {center_X - w / 2}, y오차 : {center_Y - h / 2}, depth: {np.round(roi_w / w * 100, 2)}%')

        # draw roi rectangle
        cv2.rectangle(img,
                      (int(center_X - roi_w / 2), int(center_Y - roi_h / 2)),
                      (int(center_X + roi_w / 2), int(center_Y + roi_h / 2)),
                      color=color_red,
                      thickness=2)
        cv2.putText(img,
                    f'x : {center_X - w / 2}, y : {center_Y - h / 2}, depth: {np.round(roi_w / w * 100, 2)}%',
                    (int(center_X - roi_w / 2), int(center_Y - roi_h / 2)),
                    cv2.FONT_HERSHEY_PLAIN, 1, color_red)
        ROI = origin[int(center_Y - roi_h / 2):int(center_Y + roi_h / 2),
                     int(center_X - roi_w / 2):int(center_X + roi_w / 2)]
        rotated_ROI = rotate_image(ROI, landmarks=np.array([[p.x, p.y] for p in dlib_shape.parts()]))

    cv2.imshow('my webcam', img)
    cv2.imshow('roi', rotated_ROI)

    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()
