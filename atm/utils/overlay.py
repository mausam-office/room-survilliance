import logging
import cv2
import numpy as np
from utils import DetectionConfigs
# from atm.rules import rules
from utils.angle import calculate_angle


class GeometricShapes:
    """
    Plotting geometric shapes in image.
    Args:
        ignore: sequence of plot types to ignore.
    """
    def __init__(self, ignore: list|tuple = []) -> None:
        self.ignore = ignore

    # def init_rules(self):
    #     self.sitted_rule = rules.SittedRule()
    #     self.hand_contracted_rule = rules.HandContractRule()
        
    #     self.rule = rules.RuleExecuter()


    def plot(self, image, landmarks, list_of_tuples=[], *args, **kwargs):
        """
        Plotting shapes over image.
        """
        if (ignore:=kwargs.get('ignore')) is not None and isinstance(ignore, (list, tuple)):
            self.ignore = ignore
        
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expecting numpy array.")
        
        img_shape = self.image_shape(image)
        print(img_shape)
        
        if (ndim := len(img_shape)) != 3:
            raise ValueError(f"Expecting 3-D numpy array.")

        if landmarks is None:
            logging.warn(f'No Landmarks provided. Ignoring overlays.')
            return image

        if 'connection' not in self.ignore:
            image = self.__plot_pose(image, landmarks)

        if 'angle_arc' not in self.ignore:
            image = self.__plot_angle_arc(image, landmarks, list_of_tuples)

        # if 'angle' not in self.ignore:
        #     image = self.__plot_angle(image, center=(50, 50), angle=50)
        
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
    
    def __plot_angle_arc(self, image, landmarks, list_of_tuples):
        for tripoints in list_of_tuples:
            if not isinstance(tripoints, (tuple, list, set)) and not len(tripoints) in [2, 3]:
                continue
            
            points = [landmarks.landmark[point_idx] for point_idx in tripoints]
            h, w, _ = image.shape
            
            angle_degree, angle1, angle2 = calculate_angle(w, h, *points)

            if len(tripoints) == 2:
                self.__plot_vertical_line(image, points[1], w, h)
                
            # print(points[1].x, points[1].y)
            # print(f"{angle1=}, {angle2=}")
            x_c, y_c = int(points[1].x*w), int(points[1].y*h)
            image = cv2.ellipse(image, (x_c, y_c), (23, 23), 0, angle1, angle2, (255, 0, 0), thickness=2)
        return image

    
    # def __plot_angle(self, image, center, angle):
    #     x, y = center
    #     return cv2.putText(image, f"{angle:.2f}", (x+15, y-15), fontFace=1, fontScale=2, color=(0, 0, 255), thickness=1)

    def __plot_vertical_line(self, image, p2, w, h):
        x, y = int(p2.x*w), int(p2.y*h) 
        image = cv2.line(image, (x, y), (x, 0), color=(125,125,125), thickness=1)


    def image_shape(self, image):
        return image.shape
