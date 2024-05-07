

class Postprocess:
    def __init__(self, q) -> None:
        self.q = q
        super().__init__()

    def process(self, q):
        if q.qsize():
            # print("From postprocess thread: ", self.q.qsize())
            data = q.get()
            # print('----'*10)
            # TODO apply conditions based on distance calculation