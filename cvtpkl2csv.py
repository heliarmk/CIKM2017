import numpy as np
import joblib
import os
import fire

def cvt(filename):
    output = joblib.load(filename)
    pred = output["output"]
    outputdir = filename[:filename.rfind("/")+1] + "csvfile/"
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)
    csvfile = outputdir + filename[filename.rfind("/"):filename.rfind(".")+1] + "csv"
    out = np.array(pred).reshape(2000)
    np.savetxt(fname=csvfile, X=out, fmt="%.3f",delimiter="")

if __name__ == "__main__":
    fire.Fire(cvt)