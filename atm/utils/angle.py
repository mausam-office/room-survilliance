import numpy as np # type: ignore

from dataclasses import dataclass
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark # type: ignore


@dataclass
class Point:
	x = 0
	y = 0
	visibility = 0.0

def calculate_angle(p1, p2, p3=None):
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
		angle_radian = np.arctan2(p3.y-p2.y, p3.x-p2.x) - np.arctan2(p1.y-p2.y, p1.x-p2.x)
		angle_degree = np.abs(angle_radian * 180.0 / np.pi)
		# if angle_degree > 180.0:
		# 	angle_degree = 360.0 - angle_degree
	except Exception as e:
		exit()
	return (angle_degree, p3)