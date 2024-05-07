from typing import Any
import mediapipe as mp

from dataclasses import dataclass

@dataclass
class DetectionConfigs:
    MP_DRAWING = mp.solutions.drawing_utils
    MP_DRAWING_STYLES = mp.solutions.drawing_styles
    MP_POSE = mp.solutions.pose


class Detection:
    def __init__(self, callback_func) -> None:
        self.callback_func = callback_func
        self.pose = DetectionConfigs.MP_POSE.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)

    def __call__(self, image) -> Any:
        self.detect(image)

    def detect(self, image):
        # with DetectionConfigs.MP_POSE.Pose(
        #     min_detection_confidence=0.5,
		#     min_tracking_confidence=0.5) as pose:

        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        self.callback_func(results)
