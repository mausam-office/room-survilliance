import sys
import time
import cv2
from multiprocessing import Queue

from callbacks.image_storage import SetImageCallback
from callbacks.result_storage import SetResultCallback
from configs.features import COLUMNS 
from model.pipelines.prediction_pipeline import prediction_pipeline
from utils.fresh_frame import FreshestFrame
from utils.pose_detection import Detection
from utils.postprocess import Postprocess
from utils.save_dataset import CSVDataset

TF_ENABLE_ONEDNN_OPTS=0

image  = None
# video_source = "rtsp://admin:admin@192.168.1.188"
video_source = "media/27.mp4"
# model_name = 'sgd_v3_(datast_train,dataset)'
model_name = 'sgd_model'

q = Queue()

image_callback = SetImageCallback()
ploted_image_callback = SetImageCallback()
result_callback =  SetResultCallback(q)
csv_dataset = CSVDataset('./data/dataset.csv', COLUMNS)
detect = Detection(result_callback.set_result)
pp = Postprocess(q, csv_dataset)


def callback_set_image(img):
    global image
    image =  img

def register_video_source(video_source):
    if video_source.startswith('rtsp://'):
        cam = FreshestFrame(camera=video_source, callback=None)#, callback=callback_set_image)
    else:
        cam = cv2.VideoCapture(video_source)
    return cam

def call_postprocess(image):
    data = pp.process(
        image, 
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
        actions=[],
        label=None,
        unique_indices=[7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28],
        app='streamlit',
        plotted_img_callaback = None,
        parse=False
    )
    return data


def parse(unparsed_data):
    if unparsed_data is None and not isinstance(unparsed_data, tuple):
        return
    return csv_dataset.parse(*unparsed_data, label='', req_ret=True)


def pose_classify(df):
    return prediction_pipeline(df, model_name=model_name)
    
def overlay(image, predicted_pose=None):
    return cv2.putText(
        image,
        text = f'Pose detected: {predicted_pose}',
        org=(20,75),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        thickness=2,
        color=(255,0,0),
        lineType=cv2.LINE_AA
    )

# cam = FreshestFrame(camera="rtsp://192.168.10.12:8554/profile0", callback=None)#, callback=callback_set_image)
# cam = cv2.VideoCapture(0)#, callback=callback_set_image)

# fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
# out = cv2.VideoWriter('ggg.mp4', fourcc ,20, (1920, 1080)) 


if __name__ == "__main__":
    cam = register_video_source(video_source)

    try:
        while True:
            start = time.time()
            ret, image = cam.read()
            if not ret:
                break
            if image is None:
                continue

            detect(image)

            unparsed_data = call_postprocess(image)
            df = parse(unparsed_data)
            if df is None:
                continue
            predicted_pose = pose_classify(df)
            # image = overlay(image, predicted_pose)

            # cv2.imshow('frame', image)
            # # out.write(image)
            # # cv2.imwrite('sss.jpg', image)
            # # print(image.shape)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            print(f"FPS: {1/(time.time() - start)}")
    except KeyboardInterrupt:
        pass

    cam.release()
    cv2.destroyAllWindows()