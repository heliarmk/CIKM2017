# coding: utf-8
# %load convert_data_test.py
import numpy as np
import joblib
data = []
with open("train.txt") as f:
    for line in f:
        line = line.split(",")[1:]
        data_dict = {}
        data_dict["label"] = np.array(line[0], dtype="float16")
        data_dict["input"] = np.array(line[1].split(), dtype="int16").reshape(15,4,101,101)
        data.append(data_dict)
        data_dict = {}
joblib.dump(filename="train.pkl", compress=3,value=data)
