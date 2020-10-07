from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

classes = ["sana", "momo", "mina"]

num_classes = len(classes)
image_size = 50

X_train = []
Y_train = []
Y_test = []
X_test = []
num_testdata = 18


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
        if i < num_testdata:
            X_test.append(data)
            Y_test.append(index)
        else:
            X_train.append(data)
            Y_train.append(index)
            for angle in range(-20, 20, 5):
                # 回転
                img_r = image.rotate(angle)
                data = np.asarray(img_r)
                X_train.append(data)
                Y_train.append(index)

                img_trans = image.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(img_trans)
                X_train.append(data)
                Y_train.append(index)
        # X.append(data)
        # Y.append(index)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(Y_train)
y_test = np.array(Y_test)
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("./twice_aug.npy", xy)