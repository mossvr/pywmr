
import usb1
import numpy as np
import time
import struct
import sys
import io
import binascii
from enum import Enum
import threading

MICROSOFT_VID =                 0x045e
HOLOLENS_SENSORS_PID =          0x0659
TRANSFER_SIZE =                 0x96C00
HOLOLENS_MAGIC =                0x2B6F6C44

class WmrCamera:

    class CameraFrame:
        def __init__(self):
            self.frame_type = 0
            self.seq = 0
            self.image = np.zeros(1280*481, dtype=np.uint8)

    def __init__(self):
        self.started = False
        self.thread = None

        self.pool = [
            WmrCamera.CameraFrame(),
            WmrCamera.CameraFrame(),
            WmrCamera.CameraFrame(),
            WmrCamera.CameraFrame(),
        ]
        self.frame = self.pool.pop()
        self.frames = []

        self.cv = threading.Condition()

        self.context = usb1.USBContext()
        self.dev = self.context.openByVendorIDAndProductID(
            MICROSOFT_VID, HOLOLENS_SENSORS_PID, skip_on_error=True)

        if self.dev is None:
            raise ValueError('Device not found')

        self.dev.claimInterface(3)
        self.dev.resetDevice()

        self.transfers = [
            self.dev.getTransfer(),
            self.dev.getTransfer(),
        ]

        for transfer in self.transfers:
            transfer.setBulk(0x85, TRANSFER_SIZE, self.transfer_cb)
            transfer.setBuffer(TRANSFER_SIZE)

    def __del__(self):
        self.stop()
        self.dev.resetDevice()
        self.dev.releaseInterface(3)
        self.context.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def handle_events(self, tv=0):
        self.context.handleEventsTimeout(tv=tv)

    def thread_func(self):
        
        self.dev.bulkWrite(0x5, struct.pack('<IIHH',
            HOLOLENS_MAGIC, # magic
            0x0c, # length
            0x81, # command
            0x00, # padding maybe?
        ))

        for transfer in self.transfers:
            transfer.submit()

        while any(transfer.isSubmitted() for transfer in self.transfers):
            try:
                self.context.handleEvents()
            except usb1.USBErrorInterrupted:
                pass
        

    def start(self):
        self.thread = threading.Thread(target=self.thread_func)
        self.started = True
        self.thread.start()

    def stop(self):
        if self.started:
            self.started = False
            self.thread.join()
            self.thread = None

            self.dev.bulkWrite(0x5, struct.pack('<IIHH',
                HOLOLENS_MAGIC, # magic
                0x0c, # length
                0x82, # command
                0x00, # padding maybe?
            ))

    def set_exposure_gain(self, camera_id, exposure, gain):
        self.dev.bulkWrite(0x5, struct.pack('<IIHHHHH',
            HOLOLENS_MAGIC, # magic
            0x12, # length
            0x80, # command
            camera_id,
            exposure, # 0-49665
            gain, # 0-255
            camera_id))

    def is_running(self):
        return self.started

    def transfer_cb(self, transfer: usb1.USBTransfer):
        if not self.started:
            return

        if transfer.getStatus() == usb1.TRANSFER_COMPLETED:
            self.parse_buffer(transfer.getBuffer()[:transfer.getActualLength()])

        # read the next frame
        transfer.submit()

    def reset(self):
        restart = self.started
        self.dev.resetDevice()
        if restart:
            self.start()

    def parse_buffer(self, buffer):

        # discard partial frames
        if len(buffer) < 616538:
            return

        next_id = 0
        pos = 0

        while len(buffer) - pos >= 4:

            # read the magic
            magic, = struct.unpack('<I', buffer[pos:pos+4])
            if magic != HOLOLENS_MAGIC:
                break

            # check if this is the last chunk
            if next_id == 26:
                frame_type, = struct.unpack('<H', buffer[pos+4:pos+6])
                self.frame.frame_type = frame_type
                
                with self.cv:
                    if len(self.pool) != 0:
                        self.frames.append(self.frame)
                        self.frame = self.pool.pop()
                        self.cv.notify()

                break

            
            # read the header
            header = buffer[pos:pos+32]
            pos += 32

            frame_id, = struct.unpack('<I', header[4:8])
            chunk_id, = struct.unpack('<B', header[8:9])
            chunk_len = 2100 if chunk_id == 25 else 24544

            if chunk_id != next_id:
                break
            
            next_id = chunk_id + 1

            if chunk_id == 0:
                self.frame.seq = frame_id

            offset = chunk_id * 24544
            count = 2080 if chunk_id == 25 else chunk_len
            self.frame.image[offset:offset+count] = buffer[pos:pos+count]

            pos += chunk_len

    def read(self):
        self.cv.acquire()
        while len(self.frames) == 0:
            self.cv.wait()
        frame = self.frames.pop(0)
        self.cv.release()

        img = frame.image.view().reshape((481,1280)).copy()
        ftype = frame.frame_type

        with self.cv:
            self.pool.append(frame)

        return img, frame.frame_type
