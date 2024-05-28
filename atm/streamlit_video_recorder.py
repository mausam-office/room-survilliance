import time
import cv2
import streamlit as st

from utils.saving_video import save_video, video_source, get_output_path, video_writer

def record_video():
    st.session_state.ord_q = 0
    retry = 0
    max_retry = 3
    video_src = st.session_state.video_source.strip()

    output_folder = 'media/'
    op_filename = st.session_state.op_filename
    st.session_state.video_source = int(video_src) if video_src.isnumeric() else video_src
    st.session_state.op_filename = op_filename if op_filename.endswith('.mp4') else op_filename + '.mp4'

    cam = video_source(st.session_state.video_source)
    output_path = get_output_path(output_folder=output_folder, output_name=st.session_state.op_filename)
    out = video_writer(st.session_state.video_source, output_path)

    vid_duration = st.session_state.vid_duration*60

    st.session_state.progress_placeholder.progress(0.0, text="Recording in action.")

    start_time = time.time()
    
    while True:
        elapsed_time = time.time() - start_time
        st.session_state.progress_placeholder.progress(min(1.0 * elapsed_time / vid_duration, 1.0), text="Recording in action.")
        if elapsed_time >= vid_duration:
            print("recording limit reached")
            break

        ret, frame = cam.read()   
        if not ret:
            retry += 1
            if retry > max_retry:
                break
            continue
        
        st.session_state.ip_camera_input_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        out.write(frame)

    cam.release()
    out.release()
    cv2.destroyAllWindows()

if 'ip_camera_input_placeholder' not in st.session_state:
    st.session_state.ip_camera_input_placeholder = st.empty()

if 'progress_placeholder' not in st.session_state:
    st.session_state.progress_placeholder = st.empty()

with st.sidebar:
    st.write('Video Recording')
    st.session_state.video_source = st.text_input('Video Source')
    st.session_state.op_filename = st.text_input('Output filename')
    st.session_state.vid_duration =  st.slider("Enter Video Duration (Min)", 0.5, 100.0, 0.5, 0.5)
    st.button("Start Recording", on_click=record_video)



# st.session_state.ip_camera_input_placeholder.image("Camera Input.")