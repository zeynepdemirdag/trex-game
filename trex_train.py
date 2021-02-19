import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

imgs = glob.glob("./img_final/*.png")

width = 125
height = 50

X = []
Y = []

for img in imgs:
    fileName = os.path.basename(img)
    label = fileName.split("_")[0]

    im = np.array(Image.open(img).convert("L").resize((width, height)))
    im = im // 255  # normalize
    X.append(im)
    Y.append(label)

X = np.array(X)
X = X.reshape(X.shape[0], width, height, 1)


# sns.countplot(Y)
# plt.show()


def oneHotLabels(values):
    labelEncoder = LabelEncoder()

    integer_encoded = labelEncoder.fit_transform(values)
    oneHotEncoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    oneHot_encoded = oneHotEncoder.fit_transform(integer_encoded)
    return oneHot_encoded


Y = oneHotLabels(Y)

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.25, random_state=2)

# CNN Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(width, height, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())

# classification
model.add(Dense(128, activation="relu"))
model.add(Dense(3, activation="softmax"))

# if os.path.exists("./trex_weight.h5"):
#     model.load_weights("trex_weight.h5")
#     print("Weights are loaded...")


# parameters
model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

# train
model.fit(train_X, train_Y, epochs=35, batch_size=64)

score_Train = model.evaluate(train_X, train_Y)
print("Train Accuracy: %", score_Train[1]*100)

score_Text = model.evaluate(test_X,test_Y)
print("Test Accuracy: %", score_Text[1]*100)

open("model_last.json", "w").write(model.to_json())
model.save_weights("trex_weight_last.h5")



