
import usb1
import numpy as np
import time
import struct
import sys
import io
from enum import Enum

MICROSOFT_VID =                 0x045e
HOLOLENS_SENSORS_PID =          0x0659
TRANSFER_SIZE =                 617472

class WmrCamera:

    def __init__(self, cb=None):
        self.cb = cb
        self.started = False
        self.frame_timer = 0
        self.seq = 0
        self.image = np.zeros(1280*481, dtype=np.uint8)
        self.chunk = io.BytesIO(bytearray(24576))
        self.chunk_id = None
        self.chunk_len = 0
        self.off = 0

        self.context = usb1.USBContext()
        self.dev = self.context.openByVendorIDAndProductID(
            MICROSOFT_VID, HOLOLENS_SENSORS_PID, skip_on_error=True)

        if self.dev is None:
            raise ValueError('Device not found')

        self.dev.claimInterface(3)
        self.dev.resetDevice()

        self.transfer = self.dev.getTransfer()
        self.transfer.setBulk(0x85, TRANSFER_SIZE, self.transfer_cb)
        self.transfer.setBuffer(TRANSFER_SIZE)

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

    def start(self):
        self.started = True
        self.frame_timer = time.time()
        self.dev.bulkWrite(0x5, [0x44, 0x6c, 0x6f, 0x2b, 0x0c, 0x00, 0x00, 0x00, 0x81, 0x00, 0x00, 0x00])
        self.transfer.submit()

    def stop(self):
        self.started = False
        if self.transfer.isSubmitted():
            self.transfer.cancel()
        self.dev.bulkWrite(0x5, [0x44, 0x6c, 0x6f, 0x2b, 0x0c, 0x00, 0x00, 0x00, 0x81, 0x00, 0x00, 0x00])

    def set_exposure_gain(self, camera_id, exposure, gain):
        self.dev.bulkWrite(0x5, struct.pack('<10BHHHH',
            0x44, 0x6c, 0x6f, 0x2b, 0x12, 0x00, 0x00, 0x00, 0x80, 0x00,
            camera_id, gain, exposure, camera_id))

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
    
    def parse_chunk(self, chunk):
        header = chunk[0:32]
        magic, = struct.unpack('<L', header[0:4])
        if magic != 0x2B6F6C44:
            print('wrong magic')
            return

        frame, = struct.unpack('<H', header[4:6])
        chunk_id, = struct.unpack('<B', header[8:9])

        if chunk_id == 25:
            seq, = struct.unpack('<B', self.image[89:90])
            if self.cb is not None:
                self.cb(self, seq)
        else:
            size = 24576-32
            offset = chunk_id * size
            self.image[offset:offset+size] = chunk[32:]

    def parse_buffer(self, buffer):
        pos = 0

        while pos < len(buffer):
            if self.chunk_id == None:
                header = buffer[pos:pos+32]
                magic, = struct.unpack('<L', header[0:4])

                if magic == 0x2B6F6C44:
                    self.off = 0
                    self.chunk_id, = struct.unpack('<B', header[8:9])
                    self.chunk_len = 2138 if self.chunk_id == 25 else 24576
                    self.chunk.seek(0)
                else:
                    print('out of sync')
                    return
            else:
                remaining = len(buffer) - pos
                count = min(remaining, self.chunk_len - self.chunk.tell())
                self.chunk.write(buffer[pos:pos+count])
                pos += count

                if self.chunk.tell() == self.chunk_len:
                    self.parse_chunk(self.chunk.getbuffer())
                    self.chunk.seek(0)
                    self.chunk_id = None
                    self.chunk_len = 0


    def get_image(self, id):
        return self.image.view()[1280:].reshape((480,1280))[0:480]

    def get_meta(self, id):
        img = self.image.view()

        if (id & 1) == 0:
            return img[0:640]
        
        return img[640:1280]
