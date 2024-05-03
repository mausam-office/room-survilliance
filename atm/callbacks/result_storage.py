from utils import POSE_LANDMARKS_NAMES


class SetResultCallback:
    def __init__(self, q) -> None:
        self.q = q

    def set_result(self, result):
        if result.pose_landmarks is not None:
            landmarks = {
                POSE_LANDMARKS_NAMES[idx]:lm 
                for idx, lm in enumerate(result.pose_landmarks.landmark)
            }
        else:
            landmarks = {}
        self.q.put(landmarks)
