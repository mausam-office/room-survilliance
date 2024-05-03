import math
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark # type: ignore


class Distance:
	def __init__(self) -> None:
		self.methods = ['euclidian', 'manhatten']

	def __call__(self, method, *args, **kwargs):
		assert len(args) == 2, "Two arguments in necessary."
		for arg in args:
			assert isinstance(arg, NormalizedLandmark), f"Type mismatched. Must be mediapipe.framework.formats.landmark_pb2.NormalizedLandmark"

		assert method in self.methods, f"Distance calculation method: {method} is not available."

		p1, p2 = args[0], args[1]
		
		match method:
			case 'euclidian':
				return self.euclidian_distance(p1, p2)

			case 'manhatten':
				return self.manhatten_distance(p1, p2)

	def euclidian_distance(self, p1, p2):
		# return math.sqrt(
		# 	(p2.x-p1.x) ** 2 + (p2.y-p1.y) ** 2
		# )
		return math.hypot(p2.x-p1.x, p2.y-p1.y)

	def manhatten_distance(self, p1, p2):
		raise NotImplementedError("Manhatten Distance calculation not implemented.")
