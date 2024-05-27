import numpy as np # type: ignore

from dataclasses import dataclass
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark # type: ignore


@dataclass
class Point:
	x = 0
	y = 0
	visibility = 0.0

def calculate_angle(w, h, vertex_pnt_idx, p1, p2, p3=None):
	"""
	Calculates the angle between the lines.
	Args:
        p1: Point 1 of one line
		p2: Intersection point of line 1 and line 2
		p3: [Optional] If not provided, New point on x-axis with x-coordinate being of p2.
	Returns:
        float: (angle between lines.)
	"""
	assert isinstance(p1, NormalizedLandmark), f"Not a valid point type: {p1}"
	assert isinstance(p2, NormalizedLandmark), f"Not a valid point type: {p2}"
	assert isinstance(p3, (NormalizedLandmark)) or p3 is None, f"Not a valid point type: {p3}"
	
	if p3 is None:
		p3 = Point()
		p3.x = p2.x
		p3.y = 0
		p3.visibility = 1.0

	try:  
		# angle_radian = np.arctan2(p3.y-p2.y, p3.x-p2.x) - np.arctan2(p1.y-p2.y, p1.x-p2.x)
		# angle_degree = np.abs(angle_radian * 180.0 / np.pi)
		# if angle_degree > 180.0:
		# 	angle_degree = 360.0 - angle_degree
		start_angle = np.arctan2(int(p1.y*h) - int(p2.y*h), int(p1.x*w) - int(p2.x*w)) * 180 / np.pi
		end_angle = np.arctan2(int(p3.y*h) - int(p2.y*h),  int(p3.x*w) - int(p2.x*w)) * 180 / np.pi

		if vertex_pnt_idx % 2 == 0:
			'Even points greater than 12 are at right below neck'
			# arc_angle1 = arc_angle1 if arc_angle1<0 else arc_angle1-360
			# angle = angle if arc_angle1<-90 and arc_angle1>-180 else 360-angle
			if end_angle < start_angle:
				start_angle -= 360
		else:
			if end_angle > start_angle:
				end_angle -= 360
		angle_degree = abs(start_angle-end_angle) #if p1.visibility > 0.75 and p2.visibility > 0.75 and p3.visibility > 0.75 else -1

	except Exception as e:
		raise e
	return (angle_degree, start_angle, end_angle)