from __future__ import print_function

import cv2
import numpy as np
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

video_source = st.text_input("Enter video source: ").strip()

def get_video_source():
    if len(video_source) < 1:
        st.stop()

    st.session_state.video_source = int(video_source) if video_source.isnumeric() else video_source

def change_video_source():
    del st.session_state['cam']
    get_video_source()

def set_video_source():
    if 'cam' not in st.session_state:
        if MODEL_OPS:
            print("Here in model ops")
            st.session_state['cam'] = FreshestFrame(
                camera="rtsp://192.168.10.12:8554/profile0", 
                callback=st.session_state['image_callback'].set_image
            )
        else:
            st.session_state['cam'] = cv2.VideoCapture(st.session_state.video_source)
        
        if st.session_state.get('cam') is not None:
            st.session_state.selected_frame_num = st.slider(
                "Choose Frame Number", 
                0, 
                (total_frames:=int(st.session_state['cam'].get(cv2.CAP_PROP_FRAME_COUNT))-3), 
                0
            )

get_video_source()

set_video_source()

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

# if 'cam' not in st.session_state:
#     if MODEL_OPS:
#         print("Here in model ops")
#         st.session_state['cam'] = FreshestFrame(
#             camera="rtsp://192.168.10.12:8554/profile0", 
#             callback=st.session_state['image_callback'].set_image
#         )
#     else:
#         st.session_state['cam'] = cv2.VideoCapture(st.session_state.video_source)

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

if 'label' not in st.session_state:
    st.session_state['label'] = 'standing'

if 'btn_Next' not in st.session_state:
    st.session_state['btn_Next'] = False

if 'btn_Prev' not in st.session_state:
    st.session_state['btn_Prev'] = False

if 'selected_frame_num' not in st.session_state:
    st.session_state['selected_frame_num'] = 0

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
        st.session_state['csv_dataset'].parse(*st.session_state.unparsed_data, st.session_state.label)


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

def transform_image(key):
    img=cv2.cvtColor(st.session_state[key].image, cv2.COLOR_BGR2RGB)
    return cv2.resize(img,None,fx=0.75,fy=0.75)


def set_image():
    # print(f"index Before setting: {st.session_state['cam'].get(cv2.CAP_PROP_POS_FRAMES)}")
    success, image = st.session_state['cam'].read()
    if not success:
        st.toast("Failed to retrieve frame from video.")
        image = None
    st.session_state['image_callback'].set_image(image)

def set_clicked(key):
    st.session_state[key] = True

@st.experimental_fragment
def labels_widget():
    st.session_state.label = st.radio('Pose Selection', ['standing', 'sitted', 'bowed'])

def prev_frame():
    current_frame_pos = st.session_state['cam'].get(cv2.CAP_PROP_POS_FRAMES)
    previous_frame_pos = current_frame_pos - 4  # 3 for recalling last frame as it forwards by 2 frame on every rerun
    # print(f"{current_frame_pos=}, {previous_frame_pos=}")

    if previous_frame_pos >= 0:
        st.session_state['cam'].set(cv2.CAP_PROP_POS_FRAMES, previous_frame_pos)
    # print(f"Current changed : {st.session_state['cam'].get(cv2.CAP_PROP_POS_FRAMES)}")
    
    set_clicked('btn_Prev')

@st.experimental_fragment
def choose_frame_number():
    if st.session_state.get('cam') is not None:
        st.session_state.selected_frame_num = st.slider(
            "Choose Frame Number", 
            0, 
            (total_frames:=int(st.session_state['cam'].get(cv2.CAP_PROP_FRAME_COUNT))-3), 
            0 if st.session_state.selected_frame_num < 0 else min(st.session_state.selected_frame_num, total_frames)
        )

def update_frame():
    if st.session_state.get('cam') is not None:
        st.session_state['cam'].set(cv2.CAP_PROP_POS_FRAMES, st.session_state.selected_frame_num)

if FIRST_RUN and not MODEL_OPS:
    set_image()
    FIRST_RUN = False

with st.sidebar:
    labels_widget()

    st.write('---')
    st.write('Data Recording')
    st.button('Start', on_click=start_collecting)
    st.button('Stop', on_click=stop_collecting)

    st.write('---')
    st.write('Data Storing')
    st.button('Save', on_click=save)
    # st.button('Clear', on_click=clear)

    st.write('---')
    st.write('Change Video Source')
    st.button('Change', on_click=change_video_source)
    
    st.write('---')
    st.write('Fast-Forward Videos')
    # choose_frame_number()
    st.button('Update', on_click=update_frame)
        


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
    pagination_containter = st.container()
    prev_col, next_col = pagination_containter.columns(2)
    try:
        with next_col:
            st.button("Next", on_click=set_clicked('btn_Next'))
        with prev_col:
            st.button("Previous", on_click=prev_frame)
        if not st.session_state['btn_Next'] and not st.session_state['btn_Prev']:
            st.stop()

        print(f"Before clicked {st.session_state.selected_frame_num}")
        if st.session_state['btn_Prev']:
            print('prev clicked')
            st.session_state.selected_frame_num -= 1 
            if st.session_state.selected_frame_num < 0:
                st.session_state.selected_frame_num = 0
        else:
            print('next clicked')
            st.session_state.selected_frame_num += 1
        
        print(f"After clicked {st.session_state.selected_frame_num}")
        choose_frame_number()

        set_image()

        if st.session_state['image_callback'].image is None:
            st.stop()
        
        st.session_state['detect'](st.session_state['image_callback'].image)

        data = call_postprocess(st.session_state['pp'])
        st.session_state.unparsed_data = data
        st.session_state['btn_Next'] = False
        st.session_state['btn_Prev'] = False


        if st.session_state['ploted_image_callback'].image is not None:
            image_loc.image(
                transform_image('ploted_image_callback')
                # cv2.cvtColor(st.session_state['ploted_image_callback'].image, cv2.COLOR_BGR2RGB)
            )
            st.session_state['ploted_image_callback'].image = None
        else:
            image_loc.image(
                transform_image('image_callback')
                # cv2.cvtColor(st.session_state['image_callback'].image, cv2.COLOR_BGR2RGB)
            )
        st.session_state['image_callback'].image = None


        # while True:
        #     if MODEL_OPS:
        #         if st.session_state['image_callback'].image is None:
        #             continue
                
        #         # print(type(st.session_state['image_callback'].image))
        #         st.session_state['detect'](st.session_state['image_callback'].image)
        #         call_postprocess(st.session_state['pp'])

        #     else: # Data collection
        #         if st.session_state['btn_Next']:
        #             set_image()

        #         if st.session_state['image_callback'].image is None:
        #             continue
                
        #         st.session_state['detect'](st.session_state['image_callback'].image)

        #         data = call_postprocess(st.session_state['pp'])
        #         st.session_state.unparsed_data = data
        #         st.session_state['btn_Next'] = False
                           
        #     st.session_state['image_callback'].image = None

        #     if st.session_state['ploted_image_callback'].image is not None:
        #         image_loc.image(
        #             cv2.cvtColor(st.session_state['ploted_image_callback'].image, cv2.COLOR_BGR2RGB)
        #         )

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
        







