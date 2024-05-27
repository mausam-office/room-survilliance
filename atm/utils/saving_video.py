from utils.fresh_frame import FreshestFrame
import cv2
import os

def save_video(camera,output_folder,output_name):
    os.makedirs(output_folder,exist_ok=True) 
    print(output_folder,output_name)
    output_name = os.path.join(output_folder,output_name)
    # print
    cam = FreshestFrame(camera=camera , callback=None)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_name, fourcc ,50, (640, 480)) 
    while True:
        ret,frame=cam.read()   
        if ret:
            out.write(frame)
            cv2.imshow('video',frame)
            if cv2.waitKey(1) & 0xFF == ord('a'):
                print('saving video')
                break

    cam.release()
    out.release()
    cv2.destroyAllWindows()
	
