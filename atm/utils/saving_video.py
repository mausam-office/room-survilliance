import os

import cv2

from utils.fresh_frame import FreshestFrame

max_retry=3
retry=0

def save_video(camera,output_folder,output_name):
    output_path=output_path(output_folder,output_name)
    cam,out=video_writer(camera,output_path)
    while True:
        ret, frame = cam.read()   
        while (retry<max_retry):
            if not ret:
                retry += 1
            break

        out.write(frame)
        cv2.imshow('video',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('saving video')
            break

    cam.release()
    out.release()
    cv2.destroyAllWindows()
	
def output_path(output_folder,output_name):
    os.makedirs(output_folder,exist_ok=True)
    return os.path.join(output_folder,output_name)

def video_writer(camera,output_path):
    cam = FreshestFrame(camera=camera , callback=None)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc ,50, (frame_width, frame_height)) 
    return cam,out


