import math
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark # type: ignore
from utils.transform_points import MidPoint


class Distance:
	def __init__(self) -> None:
		self.methods = ['euclidian', 'manhatten']
		self.midpoint = MidPoint()

	def __call__(self, method, w, h, calc_h=False, calc_w=False, *args, **kwargs):
		assert not (calc_w and calc_h), "Both `calc_h` and `calc_w` can't be `True` "
		assert len(args) == 2, f"Two arguments in necessary but has {len(args)}."
		for arg in args:
			assert isinstance(arg, NormalizedLandmark) or isinstance(arg, tuple), f"Type mismatched. Must be mediapipe.framework.formats.landmark_pb2.NormalizedLandmark or tuple of them"

		assert method in self.methods, f"Distance calculation method: {method} is not available."

		p1, p2 = args[0], args[1]
		
		match method:
			case 'euclidian':
				return self.euclidian_distance(p1, p2, w, h, calc_h, calc_w)

			case 'manhatten':
				return self.manhatten_distance(p1, p2, w, h)

	def euclidian_distance(self, p1, p2, w, h, calc_h, calc_w):
		if isinstance(p1, NormalizedLandmark) and isinstance(p2, NormalizedLandmark):
			d = math.hypot(
				0 if calc_h else p2.x*w-p1.x*w, 
				0 if calc_w else p2.y*h-p1.y*h
			)
			midpoint = self.midpoint.transform(p1, p2, w, h)

		elif isinstance(p1, tuple) and isinstance(p2, tuple) and len(p1)==len(p2)==2:
			d1 = math.hypot(p2[0].x*w-p1[0].x*w, p2[0].y*h-p1[0].y*h)
			d2 = math.hypot(p2[1].x*w-p1[1].x*w, p2[1].y*h-p1[1].y*h)
			d = (d1+d2) / 2		# average distance 

			mp1 = self.midpoint.transform(p1[0], p2[0], w, h)
			mp2 = self.midpoint.transform(p1[1], p2[1], w, h)
			midpoint = int((mp1[0] + mp2[0]) / 2), int((mp1[1] + mp2[1]) / 2)
		else:
			midpoint = (0,0)
			d = 0

		# are_points_visible = p1.visibility > 0.75 and p2.visibility > 0.75

		return (midpoint, d) #if are_points_visible else ((0,0), 0)

	def manhatten_distance(self, p1, p2, w, h):
		raise NotImplementedError("Manhatten Distance calculation not implemented.")


calculate_distance = Distance()