import cv2
from utils import DetectionConfigs


class GeometricShapes:
    """
    Plotting geometric shapes in image.
    Args:
        ignore: sequence of plot types to ignore.
    """
    def __init__(self, ignore: list|tuple = []) -> None:
        self.ignore = ignore

    def plot(self, image, landmarks, *args, **kwargs):
        """
        Plotting shapes over image.
        """
        if (ignore:=kwargs.get('ignore')) is not None and isinstance(ignore, (list, tuple)):
            self.ignore = ignore

        if landmarks is None:
            return image

        if 'connection' not in self.ignore:
            image = self.__plot_pose(image, landmarks)
        return image
    
    def __plot_pose(self, image, landmarks):
        DetectionConfigs.MP_DRAWING.draw_landmarks(
            image,
            landmarks,
            DetectionConfigs.MP_POSE.POSE_CONNECTIONS,
            landmark_drawing_spec=DetectionConfigs.MP_DRAWING_STYLES
            .get_default_pose_landmarks_style()
        )
        return image