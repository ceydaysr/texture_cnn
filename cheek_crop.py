import os
import numpy as np
from PIL import Image

try:
    import mediapipe as mp
except ImportError:
    mp = None

LEFT_CHEEK_LANDMARKS = [234, 93, 132, 58, 172, 136, 150, 149, 176]
RIGHT_CHEEK_LANDMARKS = [454, 323, 361, 288, 397, 365, 379, 378, 400]


def _bbox_from_landmarks(landmarks, indices, width, height, pad):
    xs = [landmarks[i].x * width for i in indices]
    ys = [landmarks[i].y * height for i in indices]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad_x = (x1 - x0) * pad
    pad_y = (y1 - y0) * pad
    x0 = max(0, int(x0 - pad_x))
    y0 = max(0, int(y0 - pad_y))
    x1 = min(width, int(x1 + pad_x))
    y1 = min(height, int(y1 + pad_y))
    return x0, y0, x1, y1


def _crop_resize(image, box, out_size):
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return None
    crop = Image.fromarray(image).crop((x0, y0, x1, y1))
    crop = crop.resize((out_size, out_size), Image.BILINEAR)
    arr = np.asarray(crop, dtype=np.float32) / 255.0
    return arr


class CheekCropper:
    def __init__(self, img_size=224, pad=0.15, model_path=""):
        if mp is None:
            raise RuntimeError("mediapipe is not installed. pip install mediapipe")
        self.img_size = img_size
        self.pad = pad
        self.face_mesh = None
        self.landmarker = None

        if hasattr(mp, "solutions"):
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
            )
        else:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            if not model_path:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(base_dir, "models", "face_landmarker.task")
            if not os.path.exists(model_path):
                raise RuntimeError(
                    "Face landmarker model not found. Download it to "
                    f"'{model_path}' or pass --landmark_model."
                )
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
            self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def close(self):
        if self.face_mesh:
            self.face_mesh.close()
            self.face_mesh = None
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None

    def extract(self, image):
        if image is None:
            return None
        height, width = image.shape[:2]
        if self.face_mesh:
            results = self.face_mesh.process(image)
            if not results.multi_face_landmarks:
                return None
            landmarks = results.multi_face_landmarks[0].landmark
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.landmarker.detect(mp_image)
            if not results.face_landmarks:
                return None
            landmarks = results.face_landmarks[0]
        left_box = _bbox_from_landmarks(landmarks, LEFT_CHEEK_LANDMARKS, width, height, self.pad)
        right_box = _bbox_from_landmarks(
            landmarks, RIGHT_CHEEK_LANDMARKS, width, height, self.pad
        )
        left = _crop_resize(image, left_box, self.img_size)
        right = _crop_resize(image, right_box, self.img_size)
        if left is None or right is None:
            return None
        return [left, right]
