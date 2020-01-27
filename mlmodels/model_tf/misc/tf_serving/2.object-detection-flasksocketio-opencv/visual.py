import base64
import io
import time
from threading import Thread, ThreadError

import numpy as np
from PIL import Image

import cv2
from socketIO_client import BaseNamespace, SocketIO

img_np = None
socketIO = SocketIO("http://192.168.0.1", 8020)
live_namespace = socketIO.define(BaseNamespace, "/live")


def receive_events_thread():
    socketIO.wait()


def on_camera_response(*args):
    global img_np
    img_bytes = base64.b64decode(args[0]["data"])
    img_np = np.array(Image.open(io.BytesIO(img_bytes)))
    print("done read", time.ctime())


def run_cam():
    global img_np
    while True:
        try:
            cv2.imshow("cam2", img_np)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
        except:
            continue


live_namespace.on("camera_update", on_camera_response)
receive_events_thread = Thread(target=receive_events_thread)
receive_cam_thread = Thread(target=run_cam)
receive_events_thread.daemon = True
receive_events_thread.start()
receive_cam_thread.daemon = True
receive_cam_thread.start()
socketIO.wait()
