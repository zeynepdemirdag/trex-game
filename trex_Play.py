# import necessary libraries

from keras.models import model_from_json
import numpy as np
from PIL import Image
import keyboard
import time
from mss import mss

mon = {
    "top": 300,
    "left": 770,
    "width": 250,
    "height": 100
}
sct = mss()

width = 125
height = 50

# import model
model = model_from_json(open("model_last.json", "r").read())
model.load_weights("trex_weight_last.h5")

# down = 0, up = 1, right = 2
labels = ["Down", "Right", "Up"]
frame_rate_time = time.time()

counter = 0
ii = 0
delay = 0.4
key_down_pressed = False

while True:
    img = sct.grab(mon)
    im = Image.frombytes("RGB", img.size, img.rgb)
    im_2 = np.array(im.convert("L").resize((width, height)))
    im_2 = im_2 // 255  # normalize

    X = np.array([im_2])
    X = X.reshape(X.shape[0], width, height, 1)
    res = model.predict(X)
    result = np.argmax(res)

    if result == 0:  # down = 0
        keyboard.press(keyboard.KEY_DOWN)
        key_down_pressed = True
    elif result == 2:  # UP = 2
        if key_down_pressed:
            keyboard.release(keyboard.KEY_DOWN)

        time.sleep(delay)
        keyboard.press(keyboard.KEY_UP)

        if ii < 1500:
            time.sleep(0.3)
        elif 1500 < ii < 5000:
            time.sleep(0.2)
        else:
            time.sleep(0.17)

        keyboard.press(keyboard.KEY_DOWN)
        keyboard.release(keyboard.KEY_DOWN)
    counter += 1

    if (time.time() - frame_rate_time) > 1:
        counter = 0
        frame_rate_time = time.time()

        if ii <= 1500:
            delay -= 0.003
        else:
            delay -= 0.005

        if delay < 0:
            delay = 0

        print("--------------------")
        print("Down : {} \nRight: {} \nUp: {}".format(res[0][0], res[0][1], res[0][2]))
