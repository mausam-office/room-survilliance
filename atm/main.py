from __future__ import print_function

from multiprocessing import Queue
import time

from callbacks.image_storage import SetImageCallback
from callbacks.result_storage import SetResultCallback
from utils.fresh_frame import FreshestFrame
from utils.pose_detection import Detection
from utils.postprocess import Postprocess


def main():
    q = Queue()

    image_callback = SetImageCallback()
    result_callback = SetResultCallback(q)
    
    cam = FreshestFrame(camera=0, callback=image_callback.set_image)
    detect = Detection(result_callback.set_result)

    pp = Postprocess(q)
    try:
        while True:
            if image_callback.image is not None:
                detect(image_callback.image)
                
                pp.process(image_callback.image, q)

                image_callback.image = None
    except KeyboardInterrupt as e:
        cam.release()
        

if __name__ == "__main__":
    main()

