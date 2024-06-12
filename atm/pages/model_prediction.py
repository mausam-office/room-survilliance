import os
import time

import cv2
import streamlit as st

from multiprocessing import Queue

from callbacks.image_storage import SetImageCallback
from callbacks.result_storage import SetResultCallback
from configs.features import COLUMNS 
from configs.app_control import FIRST_RUN, MODEL_OPS
from utils.fresh_frame import FreshestFrame
from utils.pose_detection import Detection
from utils.postprocess import Postprocess
from utils.save_dataset import CSVDataset

from model.pipelines.prediction_pipeline import prediction_pipeline

st.set_page_config(layout="wide")

st.header(":blue[Model prediction]", divider='rainbow')

video_source = st.text_input("Enter video path: ").strip()

output_name =  st.text_input('Enter the output video name').strip()

def get_video_source():
    if len(video_source) < 1:
        st.stop()

    if (video_source[:5] == 'rtsp:'):
        st.session_state.video_source = video_source

    st.session_state.video_source = int(video_source) if video_source.isnumeric() else video_source


def change_video_source():
    if isinstance(st.session_state['cam'],FreshestFrame):
        st.session_state['cam'].release()
    
    del st.session_state['cam']
    get_video_source()

def set_video_source():
    if 'cam' not in st.session_state:
        if (video_source[:6] == 'rtsp:'):
            print("Here in model ops")
            st.session_state['cam'] = FreshestFrame(
                camera=st.session_state.video_source, 
                callback=st.session_state['image_callback'].set_image
            )
        else:
            st.session_state['cam'] = cv2.VideoCapture(st.session_state.video_source)
        
get_video_source()

set_video_source()



label = None
label_selected = None


model_trained = []
for path in os.listdir('atm/saved_model'):
    if os.path.isfile(os.path.join('atm/saved_model',path)):
        model_trained.append(os.path.splitext(path)[0])

def predict():
    if st.session_state['csv_dataset'] is not None :
        print('Parsing')

        if st.session_state.unparsed_data is None and not isinstance(st.session_state.unparsed_data, tuple):
            return
        df = st.session_state['csv_dataset'].parse(*st.session_state.unparsed_data, '' , req_ret=True )

        result = prediction_pipeline(df,
                                    st.session_state.ml_alogrithm.strip()
                                    )
        return result


# if 'predicted_pose' not in st.session_state:
#     st.session_state['predicted_pose'] = None

with st.sidebar:
    st.session_state.ml_alogrithm = st.radio('Select target algorithm',model_trained)

    st.write('---')

    clicked = st.button( "Predict")

    # st.write('---')

    # if clicked:
    #     predicted_pose = predict()
    #     st.session_state.predicted_pose = predicted_pose
    # def display_pose(predicted_pose):
    #     st.write('Predicted Pose:',predicted_pose)
    
    st.write('---')
    st.write('Change Video Source')
    st.button('Change', on_click=change_video_source)
    
# ## Initializing Session State variables

if 'q' not in st.session_state:
    st.session_state['q'] = Queue()

if 'image_callback' not in st.session_state:
    st.session_state['image_callback'] = SetImageCallback()

if 'ploted_image_callback' not in st.session_state:
    st.session_state['ploted_image_callback'] = SetImageCallback()

if 'result_callback' not in st.session_state:
    st.session_state['result_callback'] =  SetResultCallback(st.session_state['q'])

if 'csv_dataset' not in st.session_state:
    st.session_state['csv_dataset'] = CSVDataset('./data/dataset.csv', COLUMNS)

if 'detect' not in st.session_state:
    st.session_state['detect'] = Detection(st.session_state['result_callback'].set_result)

if 'pp' not in st.session_state:
    st.session_state['pp'] = Postprocess(
        st.session_state['q'], 
        st.session_state['csv_dataset']
    )

if 'unparsed_data' not in st.session_state:
    st.session_state['unparsed_data'] = None

if 'start' not in st.session_state:
    st.session_state['start'] = False

# if 'prediction' not in st.session_state:
#     st.session_state.prediction = False

if 'ml_alogrithm' not in st.session_state:
    st.session_state.ml_alogrithm = ''

if 'img_container' not in st.session_state: 
    st.session_state.img_container = st.empty()


def call_postprocess(pp):
    data = pp.process(
        st.session_state['image_callback'].image, 
        st.session_state['q'], 
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
        label=None,
        unique_indices=[7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28],
        app='streamlit',
        plotted_img_callaback = st.session_state['ploted_image_callback'],
        parse=True if st.session_state['start'] else False
    )

    return data

image_loc = st.empty()

# def set_image():
#     # print(f"index Before setting: {st.session_state['cam'].get(cv2.CAP_PROP_POS_FRAMES)}")
#     success, image = st.session_state['cam'].read()
#     if not success:
#         st.toast("Failed to retrieve frame from video.")
#         image = None
#     st.session_state['image_callback'].set_image(image)

# if FIRST_RUN and not MODEL_OPS:
#     set_image()
# #     FIRST_RUN = False



def video_functions(video):
#calculating the values of the video 
    # if st.session_state['cam'] is None:
    st.session_state['cam'] = cv2.VideoCapture(video)
    fps=int((st.session_state['cam'].get(cv2.CAP_PROP_FPS))/2)
    width = int(st.session_state['cam'].get(cv2.CAP_PROP_FRAME_WIDTH))
    height =int(st.session_state['cam'].get(cv2.CAP_PROP_FRAME_HEIGHT))
    VIDEO_CODEC = 'mp4v'

    print(fps)
    return fps , width , height, VIDEO_CODEC 

def transform_image(key, predicted_pose, write=False):
    img=cv2.cvtColor(st.session_state[key].image, cv2.COLOR_BGR2RGB)
    cv2.putText(
                img,
                text = f'Pose detected:{predicted_pose}',
                org=(20,75),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                thickness=2,
                color=(255,0,0),
                lineType=cv2.LINE_AA)
    
    if write:
        st.session_state.out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # out.release()
    return img

def postprocess_data():
    data = call_postprocess(st.session_state['pp'])
    st.session_state.unparsed_data = data

# if st.session_state['image_callback'].image is None:
#     st.stop()

# st.session_state['detect'](st.session_state['image_callback'].image)

def plotted_image(predicted_pose):
    if (st.session_state['image_callback'].image) is None:
        st.stop()

    if st.session_state['ploted_image_callback'].image is not None:
        image_loc.image(
            transform_image('ploted_image_callback',predicted_pose)
        )
        st.session_state['ploted_image_callback'].image = None
    else:
        image_loc.image(
            transform_image('image_callback',predicted_pose)
        )

    st.session_state['image_callback'].image = None

if 'out' not in st.session_state: 
    fps, width, height, VIDEO_CODEC = video_functions(st.session_state.video_source)
    st.session_state['out'] = cv2.VideoWriter(output_name,
                    cv2.VideoWriter_fourcc(*VIDEO_CODEC),
                    fps,
                    (width, height)
    )


def display_video(input_video):
    # if(input_video[:5] == 'rtsp:'):
    #     st.session_state['cam'] = FreshestFrame(
    #         camera=input_video, 
    #         callback=st.session_state['image_callback'].set_image
    #     )
    # else:
    #     st.session_state['cam'] = cv2.VideoCapture(st.session_state.video_source)
    
    # cam = st.session_state['cam']
    
    num_frames_skip = 2
    cnt = 0
    try:
        while True:
            start = time.time()
            success, frame = st.session_state['cam'].read()
            if not success:
                break
            cnt += 1
            if cnt < num_frames_skip:
                continue
            cnt = 0
            
            st.session_state['image_callback'].set_image(frame)
            
            st.session_state['detect'](st.session_state['image_callback'].image)

            postprocess_data()

            predicted_pose = predict()

            transform_image('image_callback', predicted_pose, write=True)
            plotted_image(predicted_pose)
            # time.sleep(0.005)
            print(f"FPS: {1/(time.time() - start)}")
            
    except KeyboardInterrupt:
        exit()


    # out.release()
    cv2.destroyAllWindows()

if clicked:
    display_video( st.session_state.video_source)

st.session_state.out.release()

st.session_state['cam'].release()
cv2.destroyAllWindows()







