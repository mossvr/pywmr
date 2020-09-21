
from wmr import WmrCamera
import threading
import numpy as np
import cv2

quit_event = threading.Event()
frame_lock = threading.Lock()
frame_event = threading.Event()
frame = np.zeros((480,1280), dtype=np.uint8)
meta = np.zeros(640, dtype=np.uint8)

def frame_cb(camera, seq):
    global frame
    global meta
    if frame_lock.acquire(blocking=False):
        meta = np.copy(camera.get_meta(0))
        frame = np.copy(camera.get_image(0))
        frame_event.set()
        frame_lock.release()


def cv_thread():
    global frame
    global meta
    img = np.zeros((480,1280,3), dtype=np.uint8)
    orb = cv2.ORB_create(nfeatures=2048)

    while not quit_event.is_set():
        if frame_event.is_set():
            frame_lock.acquire()
            frame_event.clear()
            
            img = np.copy(frame)
        
            kp = orb.detect(frame, None)
            kp, des = orb.compute(frame, kp)
            img  = cv2.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)
            frame_lock.release()

        cv2.imshow('orb', img)
        key = cv2.waitKey(10)
        if key >= 0:
            quit_event.set()


def camera_thread():
    camera = WmrCamera(frame_cb)
    camera.set_exposure_gain(0, 500, 1200) # left 1
    camera.set_exposure_gain(1, 500, 300) # right 1
    camera.set_exposure_gain(2, 500, 1200) # left 2
    camera.set_exposure_gain(3, 500, 300) # right 2
    camera.start()

    while not quit_event.is_set():
        camera.handle_events(tv=0.25)

if __name__ == '__main__':
    camera_thread = threading.Thread(target=camera_thread, name="camera")
    camera_thread.start()
    cv_thread = threading.Thread(target=cv_thread, name="cv2")
    cv_thread.start()

    camera_thread.join()
    cv_thread.join()
