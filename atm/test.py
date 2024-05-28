import cv2
from utils.fresh_frame import FreshestFrame

image  = None

def callback_set_image(img):
    global image
    image =  img


cam = FreshestFrame(camera="rtsp://192.168.1.88", callback=None)#, callback=callback_set_image)
# cam = FreshestFrame(camera="rtsp://192.168.10.12:8554/profile0", callback=None)#, callback=callback_set_image)
# cam = cv2.VideoCapture(0)#, callback=callback_set_image)

# fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
# out = cv2.VideoWriter('ggg.mp4', fourcc ,20, (1920, 1080)) 

while True:
    ret, image = cam.read()
    if not ret:
        break
    if image is None:
        continue

    cv2.imshow('frame', image)
    # out.write(image)
    cv2.imwrite('sss.jpg', image)
    print(image.shape)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()