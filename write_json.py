from turtle import shape
import torch
import torchvision
import os
import glob
import re
import numpy as np
import json

dataset_dir = "dataset_100"
fnames_txt = glob.glob(os.path.join(os.path.join(os.path.curdir, dataset_dir), "*.txt"))
fnames_png = glob.glob(os.path.join(os.path.join(os.path.curdir, dataset_dir), "*.png"))
img_names = []
for i in range (len(fnames_png)):
    if(i%2) != 0:
        continue
    name = fnames_png[i]
    img_names.append(name)
    # name = re.findall("\d{6}", name)
    # fnames_png[i] = os.path.join(os.path.join(os.path.curdir, dataset_dir), f"{name[0]}.png")

landmark = np.zeros((len(img_names), 70, 2), dtype=np.float32)
# print(len(fnames_txt))

for i in range (len(fnames_txt)):
    file = fnames_txt[i]
    with open(file, 'r') as f:
        count = 0
        for line in f.readlines():
            line = line.split('\n')[0]
            line = line.split(' ')
            landmark[i, count, 0] = np.float32(line[0])
            landmark[i, count, 1] = np.float32(line[1])
            count += 1

landmark = landmark.reshape(landmark.shape[0], -1)
data = {
        "images" : img_names,
        "landmark_localization" : landmark.tolist()}

with open("data.json", 'w') as f:
    json.dump(data, f)

