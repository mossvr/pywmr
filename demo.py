
from wmr import WmrCamera
import numpy as np
import cv2
import struct
import binascii

if __name__ == '__main__':
    camera = WmrCamera()
    camera.set_exposure_gain(0, 0, 500) # left 1
    camera.set_exposure_gain(1, 0, 500) # right 1
    camera.set_exposure_gain(2, 6000, 500) # left 2
    camera.set_exposure_gain(3, 6000, 500) # right 2
    camera.start()

    orb = cv2.ORB_create(nfeatures=2048)

    while True:
        img, frame_type = camera.read()

        # why it this wrong sometimes?
        if frame_type != 0:
            continue

        seq = img[0][89]
        exposure, = struct.unpack('>H', img[0][6:8])

        kp = orb.detect(img, None)
        kp, des = orb.compute(img, kp)
        img  = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)

        cv2.imshow('orb', img)
        key = cv2.waitKey(1)
        if key >= 0:
            break

    camera.stop()
