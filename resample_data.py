# coding: utf-8
import numpy as np
import joblib
import tensorflow as tf
import os

def sample():
    trains = joblib.load("../data/CIKM2017_train/train_Imp_3x3.pkl")
    labels = []
    for i in trains:
        labels.append(i["label"])
    
    labels = np.array(labels)
    hist, bin_edge = np.histogram(labels,bins="auto")
    bin_dict = np.digitize(labels,bin_edge)
    hist = np.append(hist[:63],[0,1])
    hist_need_num = np.max(hist) - hist
    for index, i in enumerate(hist_need_num):
        if i.item() == np.max(hist):
            hist_need_num[index] = 0

    hist_dict = {}
    for i in range(1,66):
        tmp = {i:[]}
        hist_dict.update(tmp)
    for index, i in enumerate(bin_dict):
        hist_dict[i.item()].append(index)
    
    for key in hist_dict.keys():
        hist_dict[key] = np.array(hist_dict[key],dtype=np.int16)

    """
    for index, i in enumerate(hist_need_num):
        if not i.item() == 0:
            random_sample = np.random.choice(hist_dict[index+1],size=(i.item()))
            hist_dict[index+1] = np.append(hist_dict[index+1], random_sample)
    """
    #pop zero sample class
    pop_list = []
    for key in hist_dict.keys():
        if hist_dict[key].size == 0:
            pop_list.append(key)

    for key in pop_list:
        hist_dict.pop(key)

    change_list = [43,45,46,47,54,65]
    begin_key = 42
    for key in change_list:
        hist_dict.update({begin_key:hist_dict[key]})
        hist_dict.pop(key)
        begin_key += 1

    new_datas = []
    for key in hist_dict.keys():
        print(key)
        for index in hist_dict[key]:
            data_item = trains[index.item()]
            tmp = {"input":data_item["input"],"label": data_item["label"],"categroy": key-1}
            new_datas.append(tmp)

    #new_inputs = np.array(new_inputs)
    #new_labels = np.array(new_labels)

    #joblib.dump(value=new_datas,filename="../data/CIKM2017_train/train_Imp_3x3_resampled.pkl",compress=3)
    return new_datas

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def convert_to(data_set, name):
    num_examples = len(data_set)
    for i in range(10):
        np.random.shuffle(data_set)
    filename = os.path.join("/mnt/guankai/CIKM/data/CIKM2017_train/", name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = data_set[index]["input"].astype(np.uint8).tostring()
        label_raw = data_set[index]["label"].item()
        categroy_raw = data_set[index]["categroy"]
        example = tf.train.Example(features=tf.train.Features(feature={
            'categroy':_int64_feature(categroy_raw),
            'label': _float_feature(label_raw),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

def main():
    # Get the data.
    data_set = sample()
    for i in range(10):
        np.random.shuffle(data_set)
    valid_data_num = int(len(data_set) / 10) #get 10% data for validation
    valid_set = data_set[0 : valid_data_num ]
    train_set = data_set[valid_data_num  : ]
    #testa_set = joblib.load("../data/CIKM2017_testA/testA_Imp_3x3.pkl")
    convert_to(train_set, "train_Imp_3x3_classed")
    convert_to(valid_set, "valid_Imp_3x3_classed")
    #convert_to(testa_set, "testA_Imp_3x3")
    return
if __name__ == "__main__":
    main()

