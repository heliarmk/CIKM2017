import joblib
import numpy as np

def agg(file_name,store_file):

    datas = joblib.load(file_name)
    new_datas = []

    for data in datas:
        new_datas.append(data)
        new_datas.append({"input":np.flip(data["input"],axis=2),"label":data["label"]})
        new_datas.append({"input":np.flip(data["input"],axis=3),"label":data["label"]})
        #new_datas.append({"input":np.rot90(m=data["input"],k=1,axes=(2,3)),"label":data["label"]})
        #new_datas.append({"input":np.rot90(m=data["input"],k=2,axes=(2,3)),"label":data["label"]})
        #new_datas.append({"input":np.rot90(m=data["input"],k=3,axes=(2,3)),"label":data["label"]})

    joblib.dump(value=new_datas,filename=store_file,compress=3)

if __name__ == "__main__":
    file_name = "../data//CIKM2017_train/train_Imp_3x3.pkl"
    store_file = "../data/CIKM2017_train/train_Imp_3x3_fliped.pkl"
    agg(file_name, store_file)
