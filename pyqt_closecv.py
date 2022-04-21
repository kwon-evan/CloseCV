import cv2
import sys
from close_cv import CloseCV
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui


ccv = CloseCV()


class ShowVideo(QtCore.QObject):
    flag = 0

    # ccv = CloseCV()
    camera = cv2.VideoCapture(0)

    ret, frame = camera.read()
    height, width = frame.shape[:2]

    VideoSignal1 = QtCore.pyqtSignal(QtGui.QImage)
    VideoSignal2 = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, parent=None):
        super(ShowVideo, self).__init__(parent)

    @QtCore.pyqtSlot()
    def startVideo(self):
        global frame

        run_video = True
        while run_video:
            ret, image = self.camera.read()
            color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ccv.refresh(color_swapped_image)

            try:
                # 얼굴 크기로 작은 화면의 넓이와 높이 구하기
                roi_h = ccv.face_coords[1] - ccv.face_coords[0]
                roi_w = roi_h * color_swapped_image.shape[1] / color_swapped_image.shape[0]
                # original frame 회전
                rotate_frame = self.rotate_image(color_swapped_image)
                # 작은 화면만큼 ROI 구하기
                ROI = rotate_frame[
                      int(ccv.face_center_coords[1] - roi_h / 2):int(ccv.face_center_coords[1] + roi_h / 2),
                      int(ccv.face_center_coords[0] - roi_w / 2):int(ccv.face_center_coords[0] + roi_w / 2)
                      ]
                ROI = cv2.resize(ROI, dsize=(int(color_swapped_image.shape[1] / 2), int(color_swapped_image.shape[0] / 2)),
                                 interpolation=cv2.INTER_LINEAR)
            except TypeError:
                roi_h, roi_w = int(color_swapped_image.shape[0] / 2), int(color_swapped_image.shape[1] / 2)
                ROI = color_swapped_image[0:roi_h, 0:roi_w]
                ROI = cv2.resize(ROI, dsize=(int(color_swapped_image.shape[1] / 2), int(color_swapped_image.shape[0] / 2)),
                                 interpolation=cv2.INTER_LINEAR)

            result = ccv.annotated_frame(x=True, y=True)
            # frame에 네모랑 텍스트 그리기, 목표 z 오차면 초록색 아니면 빨간색
            self.draw_rect(
                result, target_z_diff=0.25, allow_range=0.1,
                roi_w=roi_w, roi_h=roi_h
            )

            result = self.put_xyzr_text(result)

            qt_image1 = QtGui.QImage(result.data,
                                     self.width,
                                     self.height,
                                     result.strides[0],
                                     QtGui.QImage.Format_RGB888)
            self.VideoSignal1.emit(qt_image1)

            if self.flag:
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                img_canny = cv2.Canny(img_gray, 50, 100)

                img_rotate = self.rotate_image(image)

                ROI = img_rotate[
                      int(ccv.face_center_coords[1] - roi_h / 2):int(ccv.face_center_coords[1] + roi_h / 2),
                      int(ccv.face_center_coords[0] - roi_w / 2):int(ccv.face_center_coords[0] + roi_w / 2)
                      ]
                ROI = cv2.resize(ROI, dsize=(int(image.shape[1] / 2), int(image.shape[0] / 2)),
                                 interpolation=cv2.INTER_LINEAR)

                qt_image2 = QtGui.QImage(ROI.data,
                                         int(self.width / 2),
                                         int(self.height / 2),
                                         ROI.strides[0],
                                         QtGui.QImage.Format_BGR888)

                self.VideoSignal2.emit(qt_image2)

            loop = QtCore.QEventLoop()
            QtCore.QTimer.singleShot(25, loop.quit)  # 25 ms
            loop.exec_()

    @QtCore.pyqtSlot()
    def canny(self):
        self.flag = 1 - self.flag

    @QtCore.pyqtSlot()
    def draw_rect(self, image, target_z_diff, allow_range, roi_w, roi_h):
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

    @QtCore.pyqtSlot()
    def rotate_image(self, image):
        try:
            face_center = ccv.face_center_coords
            rot_mat = cv2.getRotationMatrix2D(face_center, -1 * ccv.roll_diff, 1.0)
            result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,
                                    borderValue=(255, 255, 255))
            return result
        except TypeError:
            return image

    @QtCore.pyqtSlot()
    def put_xyzr_text(self, img):
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


class ImageViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.image = QtGui.QImage()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()

    def initUI(self):
        self.setWindowTitle('Test')

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        if image.isNull():
            print("Viewer Dropped frame!")

        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    thread = QtCore.QThread()
    thread.start()
    vid = ShowVideo()
    vid.moveToThread(thread)

    image_viewer1 = ImageViewer()
    image_viewer2 = ImageViewer()

    vid.VideoSignal1.connect(image_viewer1.setImage)
    vid.VideoSignal2.connect(image_viewer2.setImage)

    push_button1 = QtWidgets.QPushButton('Start')
    push_button2 = QtWidgets.QPushButton('Canny')
    push_button1.clicked.connect(vid.startVideo)
    push_button2.clicked.connect(vid.canny)

    vertical_layout = QtWidgets.QVBoxLayout()
    horizontal_layout = QtWidgets.QHBoxLayout()
    horizontal_layout.addWidget(image_viewer1)
    horizontal_layout.addWidget(image_viewer2)
    vertical_layout.addLayout(horizontal_layout)
    vertical_layout.addWidget(push_button1)
    vertical_layout.addWidget(push_button2)

    layout_widget = QtWidgets.QWidget()
    layout_widget.setLayout(vertical_layout)

    main_window = QtWidgets.QMainWindow()
    main_window.setCentralWidget(layout_widget)
    main_window.show()
    sys.exit(app.exec_())
