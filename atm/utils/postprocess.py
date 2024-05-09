from utils.angle import calculate_angle
from utils.display import show_image
from utils.overlay import GeometricShapes
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList # type: ignore


class Postprocess:
    def __init__(self, q) -> None:
        self.q = q

    def process(self, image, q, lm_idx_list):
        if q.qsize():
            result = q.get()

            reconstructed_landmarks = self.dict_to_landmark(result)

            h, w, _ = image.shape
            lm_plot_info = []
            for tripoints in lm_idx_list:
                if not isinstance(tripoints, (tuple, list)) and not len(tripoints) in [2, 3]:
                    continue

                tripoints, action = tripoints[:-1], tripoints[-1]
                vertex_pnt_idx = tripoints[1]

                points = [reconstructed_landmarks.landmark[point_idx] for point_idx in tripoints]
                
                angle, arc_angle1, arc_angle2 = calculate_angle(w, h, vertex_pnt_idx, *points)
                x_c, y_c = int(points[1].x*w), int(points[1].y*h)

                plot_vertical_line = True if len(tripoints) == 2 else False


                lm_plot_info.append(
                    lm_info:=dict(
                        center=(x_c, y_c), 
                        angle=angle, 
                        arc_angle1=arc_angle1, 
                        arc_angle2=arc_angle2, 
                        plot_vertical_line=plot_vertical_line
                    )
                )

            image = GeometricShapes().plot(image, reconstructed_landmarks, lm_plot_info)

            show_image(image)

    def dict_to_landmark(self, result):
        landmarks = []
        if not isinstance(result, dict):
            return
        for k in range(33):
            r = result[k]
            # nl = NormalizedLandmark(x=r.x, y=r.y, z=r.z)
            landmarks.append({"x":r.x, "y":r.y, "z":r.y, "visibility":r.visibility})
        
        return NormalizedLandmarkList(landmark=landmarks)
    
    

    
    

