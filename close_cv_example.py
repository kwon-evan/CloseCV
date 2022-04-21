import cv2
import numpy as np
from close_cv import CloseCV


def put_xyzr_text(img):
    try:
        rst = img.copy()
        cv2.putText(rst,
                    f"x diff:{ccv.x_diff}",
                    (50, 50),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,
                    color=(0, 0, 255))
        cv2.putText(rst,
                    f"y diff: {ccv.y_diff}",
                    (50, 70),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,
                    color=(0, 0, 255))
        cv2.putText(rst,
                    f"z diff: {ccv.z_diff}",
                    (50, 90),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,
                    color=(0, 0, 255))
        cv2.putText(rst,
                    f"roll diff: {ccv.roll_diff}",
                    (50, 110),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,
                    color=(0, 0, 255))
        return rst
    except AttributeError:
        return img


def rotate_image(image):
    try:
        face_center = ccv.face_center_coords
        rot_mat = cv2.getRotationMatrix2D(face_center, -1 * ccv.roll_diff, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        return result
    except TypeError:
        return image


def draw_rect(image, target_z_diff, allow_range):
    _text = ""
    _color = (0, 0, 255)

    try:
        if ccv.z_diff < target_z_diff - allow_range:
            _text = "Too Far!!"
        elif target_z_diff - allow_range <= ccv.z_diff <= target_z_diff + allow_range:
            _text = "Good."
            _color = (0, 255, 0)
        elif target_z_diff + allow_range < ccv.z_diff:
            _text = "Too Close!!"
    except TypeError:
        _text = "No Face Detected."

    cv2.putText(
        image,
        _text,
        (int(ccv.face_center_coords[0] - roi_w / 2), int(ccv.face_center_coords[1] - roi_h / 2)),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=2,
        color=_color,
        thickness=2
    )
    cv2.rectangle(
        image,
        (int(ccv.face_center_coords[0] - roi_w / 2), int(ccv.face_center_coords[1] - roi_h / 2)),
        (int(ccv.face_center_coords[0] + roi_w / 2), int(ccv.face_center_coords[1] + roi_h / 2)),
        color=_color,
        thickness=2
    )


ccv = CloseCV()
webcam = cv2.VideoCapture(0)


while True:
    _, frame = webcam.read()

    # ccv 객체의 frame을 새로 써줌
    ccv.refresh(frame)
    try:
        # 얼굴 크기로 작은 화면의 넓이와 높이 구하기
        roi_h = ccv.face_coords[1] - ccv.face_coords[0]
        roi_w = roi_h * frame.shape[1] / frame.shape[0]
        # original frame 회전
        rotate_frame = rotate_image(frame)
        # 작은 화면만큼 ROI 구하기
        ROI = rotate_frame[
              int(ccv.face_center_coords[1] - roi_h / 2):int(ccv.face_center_coords[1] + roi_h / 2),
              int(ccv.face_center_coords[0] - roi_w / 2):int(ccv.face_center_coords[0] + roi_w / 2)
              ]
        ROI = cv2.resize(ROI, dsize=(int(frame.shape[1] / 2), int(frame.shape[0] / 2)), interpolation=cv2.INTER_LINEAR)
    except TypeError:
        roi_h, roi_w = int(frame.shape[0] / 2), int(frame.shape[1] / 2)
        ROI = frame[0:roi_h, 0:roi_w]
        ROI = cv2.resize(ROI, dsize=(int(frame.shape[1] / 2), int(frame.shape[0] / 2)), interpolation=cv2.INTER_LINEAR)

    close_cv = ccv.annotated_frame(x=True, y=True)
    # frame에 네모랑 텍스트 그리기, 목표 z 오차면 초록색 아니면 빨간색
    draw_rect(close_cv, target_z_diff=0.25, allow_range=0.1)
    # 화면에 텍스트 찍기
    close_cv = put_xyzr_text(close_cv)

    cv2.imshow("ORIGINAL", frame)
    cv2.imshow("Line & Log", close_cv)
    cv2.imshow("ROI", ROI)

    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
