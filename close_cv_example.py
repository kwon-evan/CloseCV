import cv2
from close_cv import CloseCV


def xyzr_text(img, closecv):
    rst = img.copy()
    cv2.putText(rst,
                f"x diff:{closecv.x_diff}",
                (50, 50),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(0, 0, 255))
    cv2.putText(rst,
                f"y diff: {closecv.y_diff}",
                (50, 70),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(0, 0, 255))
    cv2.putText(rst,
                f"z diff: {closecv.z_diff}",
                (50, 90),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(0, 0, 255))
    cv2.putText(rst,
                f"roll diff: {closecv.roll_diff}",
                (50, 110),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(0, 0, 255))
    return rst


ccv = CloseCV()
webcam = cv2.VideoCapture(0)

while True:
    _, frame = webcam.read()

    # ccv 객체의 frame을 새로 써줌
    ccv.refresh(frame)
    # frame에 선 그리기 (옵션 선택 가능, x 라인, y 라인, 회전 효과, 얼굴 상자)
    close_cv = ccv.annotated_frame(x=True, y=True, rotate=True, face_rect=True)
    # 화면에 텍스트 찍
    xyzr_text(close_cv, ccv)

    cv2.imshow("testing", frame)
    cv2.imshow("Close CV", close_cv)

    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
