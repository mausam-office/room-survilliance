import logging
import cv2
import numpy as np
from utils import DetectionConfigs


class GeometricShapes:
    """
    Plotting geometric shapes in image.
    Args:
        ignore: sequence of plot types to ignore.
    """
    def __init__(self, ignore: list|tuple = []) -> None:
        self.ignore = ignore

    def plot(self, image, landmarks, lm_properties={}, *args, **kwargs):
        """
        Plotting shapes over image.
        """
        if (ignore:=kwargs.get('ignore')) is not None and isinstance(ignore, (list, tuple)):
            self.ignore = ignore
        
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expecting numpy array.")
        
        img_shape = self.image_shape(image)
        
        if (ndim := len(img_shape)) != 3:
            raise ValueError(f"Expecting 3-D numpy array.")

        if landmarks is None:
            logging.warn(f'No Landmarks provided. Ignoring overlays.')
            return image

        if 'connection' not in self.ignore:
            image = self.__plot_pose(image, landmarks)

        if lm_properties:
            distances = lm_properties.pop('distances')
            lm_plot_info = lm_properties['angles'].values()

        # plot angles
        for info in lm_plot_info:
            center = info['center']
            angle = info['angle']
            arc_angle1 = info['arc_angle1']
            arc_angle2 = info['arc_angle2']
            plot_vertical_line = info['plot_vertical_line']

            if 'angle' not in self.ignore:
                image = self.__plot_angle(image, center, angle)

            if 'angle_arc' not in self.ignore:
                image = self.__plot_angle_arc(image, center, arc_angle1, arc_angle2)

            if plot_vertical_line:
                image = self.__plot_vertical_line(image, center)
        
        # plot distances
        for _, (midpoint, distance) in distances.items():
            image = self.__plot_distance(image, midpoint, distance)
        
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
    
    def __plot_angle_arc(self, image, center, arc_angle1, arc_angle2):
        return cv2.ellipse(
            image, center, (15, 15), 0, arc_angle1, arc_angle2, (255, 0, 0), thickness=1
        )
    
    def __plot_angle(self, image, center, angle):
        x, y = center
        return cv2.putText(image, f"{angle:.2f}", (x+5, y-5), fontFace=1, fontScale=1, color=(0, 0, 255), thickness=1)

    def __plot_vertical_line(self, image, center):
        x, y = center
        return cv2.line(image, (x, y), (x, 0), color=(125,125,125), thickness=1)

    def __plot_distance(self, image, midpoint, distance):
        x, y = midpoint
        # image = cv2.circle()
        return cv2.putText(image, f"{distance:.2f}", (x, y), fontFace=1, fontScale=1, color=(255, 0, 0), thickness=1)

    def image_shape(self, image):
        return image.shape
