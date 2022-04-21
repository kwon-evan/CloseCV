import math
import os
import numpy as np
import cv2
import dlib
from .face import Face


class CloseCV(object):
    """
    Hustar 4th Image Processing Project
    written by team CloseCV
    It provides x, y, z and roll diffs between camera and face
    """

    def __init__(self):
        self.frame = None
        self.x_diff = None
        self.y_diff = None
        self.z_diff = None
        self.roll_diff = None
        self.face_center_coords = None
        self.face_coords = None

        self._face_detector = dlib.get_frontal_face_detector()

        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    @property
    def diffs_calculated(self):
        """check that the diffs have been calculated."""
        try:
            self.x_diff = int(self.x_diff)
            self.y_diff = int(self.y_diff)
            self.z_diff = float(self.z_diff)
            self.roll_diff = float(self.roll_diff)
            return True
        except Exception:
            return False

    def refresh(self, frame):
        """
        Refreshes the frame and analyzes it.
            Arguments:
                frame (numpy.ndarray): The frame to analyze.
        """
        self.frame = frame
        self._analyze()

    def _analyze(self):
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)

        try:
            landmarks = self._predictor(frame, faces[0])
            cur_face = Face(frame, faces[0], landmarks)
            self.face_center_coords = cur_face.center_coords
            self.face_coords = [
                cur_face.top,
                cur_face.bottom,
                cur_face.left,
                cur_face.right
            ]
            self.x_diff = self.calc_x_diff()
            self.y_diff = self.calc_y_diff()
            self.z_diff = self.calc_z_diff(frame, faces[0])
            self.roll_diff = self.calc_roll_diff(landmarks)

        except IndexError:
            self.x_diff = None
            self.y_diff = None
            self.z_diff = None
            self.roll_diff = None

    def calc_x_diff(self):
        """Returns x-diff between face and frame"""
        return self.face_center_coords[0] - self.frame.shape[1] / 2

    def calc_y_diff(self):
        """Returns y-diff between face and frame"""
        return self.face_center_coords[1] - self.frame.shape[0] / 2

    @classmethod
    def calc_z_diff(cls, frame, face):
        """
        Returns a number between 0.0 and 1.0 that indicates
        the ratio of detected face's height to current frame's height
        """
        face_height = face.bottom() - face.top()
        frame_height = frame.shape[1]
        return face_height / frame_height

    @classmethod
    def calc_roll_diff(cls, landmarks):
        """returns angle diff between vertical of camera and vertical of detected face"""
        region = np.array([[p.x, p.y] for p in landmarks.parts()])
        tan_theta = (region[30][0] - region[27][0]) / (region[30][1] - region[27][1])
        theta = np.arctan(tan_theta)
        angle = theta * 180 / math.pi
        return angle

    def annotated_frame(self, x=False, y=False):
        frame = self.frame.copy()

        if x and y:
            if self.x_diff is not None and self.y_diff is not None:
                cv2.line(
                    frame,
                    (int(frame.shape[1] / 2), int(frame.shape[0] / 2)),
                    (int(frame.shape[1] / 2 + self.x_diff), int(frame.shape[0] / 2)),
                    color=(0, 0, 255),
                    thickness=1
                )
                cv2.line(
                    frame,
                    (int(frame.shape[1] / 2 + self.x_diff), int(frame.shape[0] / 2)),
                    (int(frame.shape[1] / 2 + self.x_diff), int(frame.shape[0] / 2 + self.y_diff)),
                    color=(255, 0, 0),
                    thickness=1
                )
        elif x:
            if self.x_diff is not None:
                cv2.line(
                    frame,
                    (int(frame.shape[1] / 2), int(frame.shape[0] / 2)),
                    (int(frame.shape[1] / 2 + self.x_diff), int(frame.shape[0] / 2)),
                    color=(0, 0, 255),
                    thickness=1
                )
        elif y:
            if self.y_diff is not None:
                cv2.line(
                    frame,
                    (int(frame.shape[1] / 2), int(frame.shape[0] / 2)),
                    (int(frame.shape[1] / 2), int(frame.shape[0] / 2 + self.y_diff)),
                    color=(0, 255, 0),
                    thickness=1
                )

        return frame
