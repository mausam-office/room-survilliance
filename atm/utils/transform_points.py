
from abc import ABC, abstractmethod


class Transform(ABC):
    @abstractmethod
    def transform(self):
        pass

class MidPoint(Transform):
    def transform(self, p1, p2, w, h):
        x1, y1 = p1.x * w, p1.y * h
        x2, y2 = p2.x * w, p2.y * h

        return int(abs(x2 + x1) / 2), int(abs(y2 + y1) / 2)
        