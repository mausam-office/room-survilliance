import streamlit as st

from utils.saving_video import save_video

def record_video():
    st.session_state.ord_q = 0
    video_source = st.session_state.video_source.strip()
    op_filename = st.session_state.op_filename
    st.session_state.video_source = int(video_source) if video_source.isnumeric() else video_source
    st.session_state.op_filename = op_filename if op_filename.endswith('.mp4') else op_filename + '.mp4'
    save_video(
        st.session_state.video_source, 
        'media/', 
        st.session_state.op_filename, 
        3, 
        st.session_state.vid_duration
    )

def stop_recording():
    st.session_state.ord_q = ord('q')

if 'ip_camera_input_placeholder' not in st.session_state:
    st.session_state.ip_camera_input_placeholder = st.empty()

with st.sidebar:
    st.write('Video Recording')
    st.session_state.video_source = st.text_input('Video Source')
    st.session_state.op_filename = st.text_input('Output filename')
    st.session_state.vid_duration =  st.slider("Enter Video Duration (Min)", 1, 100)
    st.button("Start Recording", on_click=record_video)
    # st.button("Stop Recording", on_click=stop_recording)



# st.session_state.ip_camera_input_placeholder.image("Camera Input.")