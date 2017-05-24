# coding: utf-8
import joblib
import fire
import numpy as np
import os

def slice_data(filename):
    data = joblib.load(filename=filename)
    for idx, i in enumerate(data):
        data[idx]["input"] = np.delete(data[idx]["input"],[3],axis=1)
        data[idx]["input"] = data[idx]["input"][:,:,46:55,46:55]
    name, suf = os.path.splitext(filename)
    outputfilename = name + "del_height_no.4_slice_7x7.pkl"
    joblib.dump(value=data, filename=outputfilename)

if __name__ == "__main__":
    fire.Fire()