from __future__ import print_function

import cv2
import pandas as pd
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

st.set_page_config(layout="wide")

st.header(":blue[Human Pose Data Collection]", divider='rainbow')

video_source = 0
label = None
label_selected = None

## Initializing Session State variables

if 'start' not in st.session_state:
    st.session_state['start'] = False

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

if 'cam' not in st.session_state:
    if MODEL_OPS:
        print("Here in model ops")
        st.session_state['cam'] = FreshestFrame(
            camera="rtsp://192.168.10.12:8554/profile0", 
            callback=st.session_state['image_callback'].set_image
        )
    else:
        st.session_state['cam'] = cv2.VideoCapture(video_source)

if 'detect' not in st.session_state:
    st.session_state['detect'] = Detection(st.session_state['result_callback'].set_result)

if 'pp' not in st.session_state:
    st.session_state['pp'] = Postprocess(
        st.session_state['q'], 
        st.session_state['csv_dataset']
    )
if 'fps_accumulated' not in st.session_state:
    st.session_state['fps_accumulated'] = 0

if 'iter_count' not in st.session_state:
    st.session_state['iter_count'] = 0

if 'unparsed_data' not in st.session_state:
    st.session_state['unparsed_data'] = None



##  Functions
def start_collecting():
    st.session_state['start'] = True

def stop_collecting():
    st.session_state['start'] = False

def save():
    if st.session_state['csv_dataset'] is not None:
        print('Parsing')
        if st.session_state.unparsed_data is None and not isinstance(st.session_state.unparsed_data, tuple):
            return
        st.session_state['csv_dataset'].parse(*st.session_state.unparsed_data, label)


        print('Saving ...')
        st.session_state['csv_dataset'].save()
        st.session_state.unparsed_data = None

def clear():
    if st.session_state['csv_dataset'] is not None:
        print('clearing ...')
        st.session_state['csv_dataset'].clear()

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
        label=label,
        unique_indices=[7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28],
        app='streamlit',
        plotted_img_callaback = st.session_state['ploted_image_callback'],
        parse=True if st.session_state['start'] else False
    )

    return data

def set_image():
    success, image = st.session_state['cam'].read()
    if not success:
        st.toast("Failed to retrieve frame from video.")
        image = None
    st.session_state['image_callback'].set_image(image)

def set_clicked(key):
    st.session_state[key] = True

@st.experimental_fragment
def labels_widget():
    label_selected = st.radio('Pose Selection', ['standing', 'sitted', 'bowed'])
    return label_selected

if FIRST_RUN and not MODEL_OPS:
    set_image()

with st.sidebar:
    label = labels_widget()

    st.write('---')
    st.write('Data Recording')
    st.button('Start', on_click=start_collecting)
    st.button('Stop', on_click=stop_collecting)

    st.write('---')
    st.write('Data Storing')
    st.button('Save', on_click=save)
    # st.button('Clear', on_click=clear)



tab_image, tab_data = st.tabs(['image', 'data'])


with tab_data:
    try:
        st.dataframe(
            pd.read_csv(
                st.session_state['csv_dataset'].filepath
            )
        )
    except Exception as e:
        st.toast(f"{e}")


with tab_image:
    image_loc = st.empty()
    try:
        st.button("Next", on_click=set_clicked('btn_Next'))
        while True:
            if MODEL_OPS:
                if st.session_state['image_callback'].image is None:
                    continue
                
                # print(type(st.session_state['image_callback'].image))
                st.session_state['detect'](st.session_state['image_callback'].image)
                call_postprocess(st.session_state['pp'])

            else: # Data collection
                if st.session_state['btn_Next']:
                    set_image()

                if st.session_state['image_callback'].image is None:
                    continue
                
                st.session_state['detect'](st.session_state['image_callback'].image)

                data = call_postprocess(st.session_state['pp'])
                st.session_state.unparsed_data = data
                st.session_state['btn_Next'] = False
                           
            st.session_state['image_callback'].image = None

            if st.session_state['ploted_image_callback'].image is not None:
                image_loc.image(
                    cv2.cvtColor(st.session_state['ploted_image_callback'].image, cv2.COLOR_BGR2RGB)
                )

    except KeyboardInterrupt as e:
        st.session_state['cam'].release()
        cv2.destroyAllWindows()


    


# try:
#     while True:
#         if image_callback.image is not None:
#             start = time.perf_counter()
#             detect(image_callback.image)
            
#             pp.process(
#                 image_callback.image, 
#                 q, 
#                 angle_calc_lm_idx_list=[
#                     # points must be in descenting fashion and point2 is central or vertex point
#                     # point1, point2, point3
#                     (16, 14, 12), 
#                     (15, 13, 11),

#                     (14, 12, 24),
#                     (13, 11, 23),

#                     (26, 24, 12),
#                     (25, 23, 11),

#                     (28, 26, 24),
#                     (27, 25, 23),
#                 ],
#                 dist_calc_lm_idx_list=[
#                     # point1, point2, name connecting keypoints with side
#                     (14, 12, 'width_elbow_shoulder_r'),
#                     (13, 11, 'width_elbow_shoulder_l'),

#                     # ((7, 8), (27, 28), 'dist_height_avg'),
#                     (8, 28, 'dist_height_r'),
#                     (7, 27, 'dist_height_l'),
#                     (13, 14, 'dist_width'),

#                     (12, 24, 'height_shoulder_waist_r'),
#                     (11, 23, 'height_shoulder_waist_l'),

#                     (24, 26, 'height_waist_knee_r'),
#                     (23, 25, 'height_waist_knee_l'),

#                     (12, 26, 'height_knee_shoulder_r'),
#                     (11, 25, 'height_knee_shoulder_l'),

#                     (24, 28, 'height_ankle_waist_r'),
#                     (23, 27, 'height_ankle_waist_l'),

#                     (11, 12, "shoulder_l_r"),
#                     (23, 24, "waist_l_r"),
#                     (25, 26, "knee_l_r"),
#                 ],
#                 actions=['hand_contraction', 'sitted'],
#                 label=label,
#                 unique_indices=[7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28],
#                 app='streamlit'
#             )

#             image_callback.image = None

#             fps_accumulated += 1/(time.perf_counter() - start)
#             iter_count += 1
# except KeyboardInterrupt as e:
#     print(f"Average FPS: {fps_accumulated/iter_count}")
#     cam.release()
#     cv2.destroyAllWindows()
        







