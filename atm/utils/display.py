import cv2

def show_image(image):
    cv2.imshow('Landmarks', image)
    cv2.waitKey(1)