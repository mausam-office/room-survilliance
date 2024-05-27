from __future__ import print_function

import atexit
import time

import cv2
from multiprocessing import Queue

from callbacks.image_storage import SetImageCallback
from callbacks.result_storage import SetResultCallback
from configs.features import COLUMNS 
from utils.fresh_frame import FreshestFrame
from utils.pose_detection import Detection
from utils.postprocess import Postprocess
from utils.save_dataset import CSVDataset


csv_dataset = None

@atexit.register
def on_exit():
    global csv_dataset

    if csv_dataset is not None:
        csv_dataset.save()

def main(label, loop=True):
    global csv_dataset
    q = Queue()

    image_callback = SetImageCallback()
    result_callback = SetResultCallback(q)

    csv_dataset = CSVDataset('./data/dataset.csv', COLUMNS)
    
    cam = FreshestFrame(camera="rtsp://192.168.10.12:8554/profile0", callback=image_callback.set_image)
    # cam = FreshestFrame(camera='D:/Anaconda/ATM Security/image data/ATM/vdo/atm-files/bharatpur.dav', callback=image_callback.set_image)
    # cam = FreshestFrame(camera=0, callback=image_callback.set_image)
    detect = Detection(result_callback.set_result)

    pp = Postprocess(q, csv_dataset)
    fps_accumulated = 0
    iter_count = 0
    try:
        while loop:
            if image_callback.image is not None:
                start = time.perf_counter()
                detect(image_callback.image)
                
                pp.process(
                    image_callback.image, 
                    q, 
                    angle_calc_lm_idx_list=[
                        # points must be in descenting fashion and point2 is central or vertex point
                        # point1, point2, point3
                        (16, 14, 12), 
                        (15, 13, 11),

                        (14, 12, 24),
                        (13, 11, 23),

                        (26, 24, 12),
                        (25, 23, 11),

                        (28, 26, 24),
                        (27, 25, 23),
                    ],
                    dist_calc_lm_idx_list=[
                        # point1, point2, name connecting keypoints with side
                        (14, 12, 'width_elbow_shoulder_r'),
                        (13, 11, 'width_elbow_shoulder_l'),

                        # ((7, 8), (27, 28), 'dist_height_avg'),
                        (8, 28, 'dist_height_r'),
                        (7, 27, 'dist_height_l'),
                        (13, 14, 'dist_width'),

                        (12, 24, 'height_shoulder_waist_r'),
                        (11, 23, 'height_shoulder_waist_l'),

                        (24, 26, 'height_waist_knee_r'),
                        (23, 25, 'height_waist_knee_l'),

                        (12, 26, 'height_knee_shoulder_r'),
                        (11, 25, 'height_knee_shoulder_l'),

                        (24, 28, 'height_ankle_waist_r'),
                        (23, 27, 'height_ankle_waist_l'),

                        (11, 12, "shoulder_l_r"),
                        (23, 24, "waist_l_r"),
                        (25, 26, "knee_l_r"),
                    ],
                    actions=['hand_contraction', 'sitted'],
                    label=label,
                    unique_indices=[7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28],
                    parse=True
                )

                image_callback.image = None

                fps_accumulated += 1/(time.perf_counter() - start)
                iter_count += 1
    except KeyboardInterrupt as e:
        print(f"Average FPS: {fps_accumulated/iter_count}")
        cam.release()
        cv2.destroyAllWindows()
        

if __name__ == "__main__":
    labels = [
        'standing',
        'bending',
        'sitted'
    ]
    main(labels[0])

