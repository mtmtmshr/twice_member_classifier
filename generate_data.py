from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

classes = ["sana", "momo", "mina"]

num_classes = len(classes)
image_size = 50

X = []
Y = []

for index, class_ in enumerate(classes):
    photos_dir = "./" + class_
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file_ in enumerate(files):
        if i >= 72:
            break
        image = Image.open(file_)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X.append(data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("./twice.npy", xy)