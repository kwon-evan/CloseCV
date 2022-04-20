import numpy as np


class Face(object):
    def __init__(self, frame, face, landmarks):
        self.landmark_points = None
        self.center_coords = None
        self.top = None
        self.bottom = None
        self.right = None
        self.left = None
        self._analyze(face, landmarks)

    def _analyze(self, face, landmarks):
        self.landmark_points = np.array([[p.x, p.y] for p in landmarks.parts()])
        self.center_coords = tuple(np.mean(self.landmark_points, axis=0).astype(np.int))
        self.top = face.top()
        self.bottom = face.bottom()
        self.right = face.right()
        self.left = face.left()
