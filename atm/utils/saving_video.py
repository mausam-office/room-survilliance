import os
import time

import cv2

from utils.fresh_frame import FreshestFrame

def get_output_path(output_folder, output_name):
    os.makedirs(output_folder, exist_ok=True)
    return os.path.join(output_folder, output_name)

def video_source(camera):
    return FreshestFrame(camera=camera , callback=None)

def video_writer(camera, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    c = cv2.VideoCapture(camera)
    frame_width = c.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = c.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out = cv2.VideoWriter(output_path, fourcc, 30, (int(frame_width), int(frame_height)))
    c.release()
    return out

def save_video(camera, output_folder, output_name, max_retry, recording_time):
    recording_time = recording_time*60
    retry = 0
    cam = video_source(camera)
    output_path = get_output_path(output_folder = output_folder, output_name = output_name)
    out = video_writer(camera, output_path)
    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time >= recording_time:
            print("recording limit reached")
            break

        ret, frame = cam.read()   
        if not ret:
            retry += 1
            if retry > max_retry:
                break
            continue

        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('saving video')
            break

    cam.release()
    out.release()
    cv2.destroyAllWindows()