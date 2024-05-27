import time
import cv2 
import threading

# also acts (partly) like a cv.VideoCapture
class FreshestFrame(threading.Thread):
	def __init__(self, *, camera, callback, queue=None, name='FreshestFrame'):
		self.queue = queue
		self.camera = camera
		self.capture = cv2.VideoCapture(self.camera)
		self.capture.set(cv2.CAP_PROP_FPS, 24)
		assert self.capture.isOpened()

		# this lets the read() method block until there's a new frame
		self.cond = threading.Condition()

		# this allows us to stop the thread gracefully
		self.running = False

		# keeping the newest frame around
		self.frame = None

		# passing a sequence number allows read() to NOT block
		# if the currently available one is exactly the one you ask for
		self.latestnum = 0

		# this is just for demo purposes		
		# self.callback = None
		self.callback = callback
		# self.event = threading.Event()
		
		super().__init__(name=name)
		self.start()

	def change_camera(self, *, camera):
		self.camera = camera
		self.capture = cv2.VideoCapture(self.camera)

		self.capture.set(cv2.CAP_PROP_FPS, 24)
		assert self.capture.isOpened()
		
	def start(self):
		self.running = True
		super().start()

	def release(self, timeout=10):
		self.running = False
		self.capture.release()
		super().join()

	def run(self):
		counter = 0
		while self.running:
			# block for fresh frame
			(rv, img) = self.capture.read()
			assert rv, exit()
			time.sleep(0.005)
			counter += 1

			# publish the frame
			with self.cond: # lock the condition for this operation
				self.frame = img if rv else None
				self.latestnum = counter
				self.cond.notify_all()

			if self.callback:
				self.callback(img)

	def read(self, wait=True, seqnumber=None, timeout=None):
		# with no arguments (wait=True), it always blocks for a fresh frame
		# with wait=False it returns the current frame immediately (polling)
		# with a seqnumber, it blocks until that frame is available (or no wait at all)
		# with timeout argument, may return an earlier frame;
		#   may even be (0,None) if nothing received yet

		# if self.event:
		# 	exit()

		with self.cond:
			if wait:
				if seqnumber is None:
					seqnumber = self.latestnum+1
				if seqnumber < 1:
					seqnumber = 1
				
				rv = self.cond.wait_for(lambda: self.latestnum >= seqnumber, timeout=timeout)
				if not rv:
					return (self.latestnum, self.frame)
			if self.queue is None:
				return (self.latestnum, self.frame)
			self.queue.put(self.frame)

