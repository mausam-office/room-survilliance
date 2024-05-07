from utils.overlay import GeometricShapes
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList # type: ignore


class Postprocess:
    def __init__(self, q) -> None:
        self.q = q

    def process(self, image, q):
        if q.qsize():
            result = q.get()

            reconstructed_landmarks = self.dict_to_landmark(result)

            image = GeometricShapes().plot(image, reconstructed_landmarks)

    def dict_to_landmark(self, result):
        landmarks = []
        if not isinstance(result, dict):
            return
        for k in range(33):
            r = result[k]
            # nl = NormalizedLandmark(x=r.x, y=r.y, z=r.z)
            landmarks.append({"x":r.x, "y":r.y, "z":r.y, "visibility":r.visibility})
        
        return NormalizedLandmarkList(landmark=landmarks)
    
    

