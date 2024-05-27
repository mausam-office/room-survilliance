import copy
import time
from configs.features import ANGLE_COLUMNS_MAPPING, COLUMNS_NAME_MAPPING
from utils.angle import calculate_angle
from utils.display import show_image
from utils.distance import calculate_distance
from utils.overlay import GeometricShapes
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList # type: ignore
# from rules import rules


class Postprocess:
    def __init__(self, q, csv_dataset) -> None:
        self.q = q
        self.csv_dataset = csv_dataset
        self.last_time_parsed = time.time()
        self.interval = 0.4

    def process(self, image, q, angle_calc_lm_idx_list, dist_calc_lm_idx_list, actions, label, unique_indices, app='', plotted_img_callaback=None, parse=True):
        if q.qsize():
            result = q.get()

            reconstructed_landmarks = self.dict_to_landmark(result)

            # print(f"{reconstructed_landmarks.landmark[26]=}\n{'----'*10}")

            h, w, _ = image.shape

            # angle calculations
            lm_properties: dict = self.get_angles(angle_calc_lm_idx_list, reconstructed_landmarks, w, h)
            
            # distance calculations
            distances = self.get_distances(dist_calc_lm_idx_list, reconstructed_landmarks, w, h)
            
            lm_properties['distances'] = distances

            visibilities = self.get_visibility(reconstructed_landmarks, unique_indices)
            lm_properties['visibilities'] = visibilities

            # print(f"{lm_properties=}\n{'---'*10}")

            image = GeometricShapes().plot(image, reconstructed_landmarks, copy.deepcopy(lm_properties))

            # rules_opted = self.get_rules(actions)
            # self.execute_rules(rules_opted, lm_properties)
            if plotted_img_callaback is not None:
                plotted_img_callaback.image = image

            if app != 'streamlit':
                show_image(image)
            if parse and (time.time()-self.last_time_parsed)>=self.interval:
                self.csv_dataset.parse(lm_properties, w, h, label)
                self.last_time_parsed = time.time()

            return lm_properties, w, h, label

    def dict_to_landmark(self, result):
        landmarks = []
        if not isinstance(result, dict):
            return
        for k in range(33):
            r = result[k]
            # nl = NormalizedLandmark(x=r.x, y=r.y, z=r.z)
            landmarks.append({"x":r.x, "y":r.y, "z":r.y, "visibility":r.visibility
                            #   , "presence":r.presence
                            })
        
        return NormalizedLandmarkList(landmark=landmarks)
    
    def get_rules(self, actions: list[str] | tuple[str]):
        rules_opted = []
        # print(f"{actions = }")
        for action in actions:
            match action:
                case 'hand_contraction':
                    rules_opted.append(rules.HandContractRule)
                
                case "sitted":
                    rules_opted.append(rules.SittedRule)
        return rules_opted
            
    def execute_rules(self, rules_opted, data):
        # print(rules_opted)
        for rule in rules_opted:
            if issubclass(rule, rules.ThreatRule):
                rules.RuleExecuter().execute(data, rule())

    def get_angles(self, angle_calc_lm_idx_list, reconstructed_landmarks, w, h):
        lm_properties = {}

        lm_properties['angles'] = {}

        for tripoints in angle_calc_lm_idx_list:
            if not isinstance(tripoints, (tuple, list)) and not len(tripoints) in [2, 3]:
                continue

            vertex_pnt_idx = tripoints[1]

            points = [reconstructed_landmarks.landmark[point_idx] for point_idx in tripoints]
            
            angle, arc_angle1, arc_angle2 = calculate_angle(w, h, vertex_pnt_idx, *points)
            x_c, y_c = int(points[1].x*w), int(points[1].y*h)

            plot_vertical_line = True if len(tripoints) == 2 else False


            lm_info = dict(
                center=(x_c, y_c), 
                angle=angle, 
                arc_angle1=arc_angle1, 
                arc_angle2=arc_angle2, 
                plot_vertical_line=plot_vertical_line
            )

            lm_properties['angles'][ANGLE_COLUMNS_MAPPING[vertex_pnt_idx]] = lm_info
            
            # self.execute_rules(self.get_rule(action), lm_info)
        return lm_properties
    
    def get_distances(self, dist_calc_lm_idx_list, reconstructed_landmarks, w, h):
        distances = {}
        for pointpair in dist_calc_lm_idx_list:
            assert len(pointpair) == 3, "Must have two point's indices and a text info."
            
            dist_name = pointpair[-1]
            pointpair = pointpair[:-1]
            
            if (
                isinstance(pp1:=pointpair[0], tuple) and 
                isinstance(pp2:=pointpair[1], tuple) and 
                len(pp1)==len(pp2)==2
            ):
                for _ in range(2):
                    points = [
                        (
                            reconstructed_landmarks.landmark[point_idx_0], 
                            reconstructed_landmarks.landmark[point_idx_1]
                        ) 
                        for point_idx_0, point_idx_1 in (pp1, pp2)
                    ]
            else:
                points = [reconstructed_landmarks.landmark[point_idx] for point_idx in pointpair]
        
            distances[dist_name] = calculate_distance(
                'euclidian', w, h,
                True if dist_name.startswith('height') else False,
                True if dist_name.startswith('width') else False,
                *points
            )

        return distances

    def get_visibility(self, reconstructed_landmarks, unique_indices):
        visibilities = {
            'visibility_' + COLUMNS_NAME_MAPPING[point_idx] : reconstructed_landmarks.landmark[point_idx].visibility
            for point_idx in unique_indices
        }
        # print(list(visibilities.keys()))
        return visibilities
