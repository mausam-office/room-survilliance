from __future__ import print_function

from multiprocessing import Queue

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

    try:
        pp = Postprocess(q)
        while True:
            if image_callback.image is not None:
                detect(image_callback.image)
                
                image_callback.image = None
                # landmarks = result_callback.result.pose_landmarks
                # pp(result_callback)
    except KeyboardInterrupt as e:
        pp.stop()
        
        cam.release()
        cam.join()
        

if __name__ == "__main__":
    main()

