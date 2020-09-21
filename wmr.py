
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

    def __init__(self):
        self.started = False
        self.thread = None

        self.pool = [
            np.zeros(1280*481, dtype=np.uint8),
            np.zeros(1280*481, dtype=np.uint8),
            np.zeros(1280*481, dtype=np.uint8),
            np.zeros(1280*481, dtype=np.uint8),
            np.zeros(1280*481, dtype=np.uint8),
        ]
        self.frame = self.pool.pop()
        self.frames = [None, None]
        self.grabbed = [None, None]

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

        self.exposure = [0,0,0,0]
        self.set_exposure_gain(0, 0, 500) # left 1
        self.set_exposure_gain(1, 0, 500) # right 1
        self.set_exposure_gain(2, 6000, 500) # left 2
        self.set_exposure_gain(3, 6000, 500) # right 2

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

        self.exposure[camera_id] = exposure // 20

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
                index = (frame_type >> 1) & 1

                exposure, = struct.unpack('>H', self.frame[6:8])

                # drop image if the exposure is wrong
                if exposure != self.exposure[frame_type]:
                    return
                
                with self.cv:
                    if len(self.pool) != 0:
                        drop_frame = self.frames[index]
                        self.frames[index] = self.frame
                        self.frame = self.pool.pop()
                        if drop_frame is not None:
                            self.pool.append(drop_frame)
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

            offset = chunk_id * 24544
            count = 2080 if chunk_id == 25 else chunk_len
            self.frame[offset:offset+count] = buffer[pos:pos+count]

            pos += chunk_len

    def grab(self):
        self.cv.acquire()
        imgs = [0, 1]

        while any(self.frames[i] is None for i in imgs):
            self.cv.wait()

        for i in imgs:
            old_grabbed = self.grabbed[i]
            self.grabbed[i] = self.frames[i]
            self.frames[i] = None
            if old_grabbed is not None:
                self.pool.append(old_grabbed)
        self.cv.release()

    def retrieve(self, id):
        index = (id>>1) & 1
        offset = (id & 1) * 640

        if self.grabbed[index] is None:
            return None

        return self.grabbed[index].view().reshape((481, 1280))[:, offset:offset+640]
