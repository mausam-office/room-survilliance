from threading import Thread


class Postprocess(Thread):
    def __init__(self, q) -> None:
        self.q = q
        super().__init__()
        self.start()

    def start(self):
        self.running = True
        super().start()

    def stop(self):
        self.running = False
        super().join()

    def run(self):
        while self.running:
            if self.q.qsize():
                data = self.q.get()
                print(data)
                print('----'*10)