from fresh_frame import FreshestFrame
import cv2

def save_video(camera,output_name):
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
	
