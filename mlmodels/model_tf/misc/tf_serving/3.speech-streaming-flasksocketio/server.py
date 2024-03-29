import base64
import random
import re
import time
from os.path import abspath, dirname
from queue import Queue

import numpy as np
from flask import Flask, Response

import speech_recognition as sr
from flask_socketio import SocketIO, emit, send

d = dirname(dirname(abspath(__file__)))
app = Flask(__name__)
app.queue = Queue()
socketio = SocketIO(app)
r = sr.Recognizer()


@socketio.on("connect", namespace="/live")
def test_connect():
    print("Client wants to connect.")
    emit("response", {"data": "OK"}, broadcast=True)


@socketio.on("disconnect", namespace="/live")
def test_disconnect():
    print("Client disconnected")


@socketio.on("event", namespace="/live")
def test_message(message):
    pass


@socketio.on("livespeech", namespace="/live")
def test_live(message):
    speech = sr.AudioData(
        base64.b64decode(message["data"]), message["sample_rate"], message["sample_width"]
    )
    value = r.recognize_google(speech, language="ms-MY")
    emit("speech_update", {"text": value}, broadcast=True)


if __name__ == "__main__":
    socketio.run(app, host="localhost", port=5000, debug=True)
