from itertools import count
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

path = os.path.join(os.path.curdir, "data")
img_path = os.path.join(path, "aflw_val")
txt_path = os.path.join(path, "prediction")
img_fname = sorted(os.listdir(img_path))
txt_fname = sorted(os.listdir(txt_path))

prediction_path = os.path.join(path, "output")
if os.path.isdir(prediction_path) is False:
    os.mkdir(prediction_path)

landmark = np.zeros((68, 2), dtype=np.int)
for i in range(len(img_fname[:10])):
    with open(os.path.join(txt_path, txt_fname[i]), "r") as f:
        count = 0
        for line in f.readlines():
            cord = line.split("\n")[0]
            cord = cord.split(" ")
            landmark[count, 0] = np.float32(cord[0])
            landmark[count, 1] = np.float32(cord[1])
            count += 1

    img = cv2.imread(os.path.join(img_path, img_fname[i + 1]))
    for point in landmark:
        cv2.circle(img, tuple(point), radius=2, color=(0, 0, 255), thickness=1)
    # plt.scatter(landmark[:,0], landmark[:,1], marker="x", color="red", s=200)
    # plt.show()
    filename = f"{img_fname[i+1]}.png"
    file = os.path.join(prediction_path, filename)
    cv2.imwrite(file, img)
