# coding: utf-8
import numpy as np
import joblib
import tensorflow as tf
import os

def sample(trains):

    labels = []
    for i in trains:
        labels.append(i["label"])
    
    labels = np.array(labels)
    hist, bin_edge = np.histogram(labels,bins="auto")
    bin_dict = np.digitize(labels,bin_edge)
    hist = np.append(hist[:63],[0,1])
    '''
    hist_need_num = np.max(hist) - hist
    for index, i in enumerate(hist_need_num):
        if i.item() == np.max(hist):
            hist_need_num[index] = 0
    '''
    #the class which its item count less than 100 would be merge in a new class
    less_than_100_list = []
    for i, item in enumerate(hist):
        if (item < 100):
            less_than_100_list.append(i + 1)
    for idx in range(bin_dict.shape[0]):
        if any([bin_dict[idx] == x for x in less_than_100_list]):
           bin_dict[idx] = 22

    hist_dict = {}

    for idx, num in enumerate(bin_dict):
        if not any(num.item() == key for key in hist_dict.keys()):
            hist_dict.update({num.item(): [idx]})
        else:
            hist_dict[num.item()].append(idx)

    hist_new = np.zeros((len(hist_dict.keys())))
    for key in hist_dict.keys():
        hist_dict[key] = np.array(hist_dict[key],dtype=np.int16)
        hist_new[key-1] = hist_dict[key].shape[0]

    """============================resample================================="""
    """
    hist_need_num = (np.max(hist_new) - hist_new).astype(np.int16)

    for index, i in enumerate(hist_need_num):
        if not i.item() == 0:
            random_sample = np.random.choice(hist_dict[index+1],size=(i.item()))
            hist_dict[index+1] = np.append(hist_dict[index+1], random_sample)
    """
    #pop zero sample class

    '''
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
    '''
    new_x = []
    new_label = []
    new_class = []
    for key in hist_dict.keys():
        print(key)
        for index in hist_dict[key]:
            data_item = trains[index.item()]
            new_x.append(data_item["input"])
            new_label.append(data_item["label"])
            new_class.append(key-1)

    #new_inputs = np.array(new_inputs)
    #new_labels = np.array(new_labels)

    #joblib.dump(value=new_datas,filename="../data/CIKM2017_train/train_Imp_3x3_resampled.pkl",compress=3)
    return np.asarray(new_x,dtype=np.int16).transpose((0,1,3,4,2)), np.asarray(new_label,dtype=np.float16).reshape(-1,1), np.asarray(new_class,dtype=np.uint8).reshape(-1,1)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def convert_to(data_set, name, is_test=False):
    num_examples = len(data_set)
    if not is_test:
        for i in range(10):
            np.random.shuffle(data_set)
        filename = os.path.join("/mnt/guankai/CIKM/data/CIKM2017_train/", name + '.tfrecords')
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(num_examples):
            #image_raw = data_set[index]["input"].astype(np.uint8).tostring()
            image_raw = data_set[index]["input"].tostring()
            label_raw = data_set[index]["label"].item()
            categroy_raw = data_set[index]["categroy"]
            example = tf.train.Example(features=tf.train.Features(feature={
                'categroy':_int64_feature(categroy_raw),
                'label': _float_feature(label_raw),
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        writer.close()
    else:
        filename = os.path.join("/mnt/guankai/CIKM/data/CIKM2017_testA/", name + '.tfrecords')
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(num_examples):
            #image_raw = data_set[index]["input"].astype(np.uint8).tostring()
            image_raw = data_set[index]["input"].tostring()
            label_raw = data_set[index]["label"].item()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _float_feature(label_raw),
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        writer.close()

def preprocessing(inputs, mean, std, is_train_set=False, only_calc_mean_std=False):
    # Zero Center the inputs
    datas = inputs
    # shape (15,101,101,4)
    #mean, var = np.moments(datas, axes=(1, 2), keep_dims=True)
    #datas = tf.subtract(datas, mean)
    if is_train_set:
        mean, std = np.zeros((15,4,1,1)), np.zeros((15,4,1,1))
        for item in datas:
            mean += np.mean(item["input"], axis=(2,3),keepdims=True)
            std += np.std(item["input"], axis=(2,3), keepdims=True)

        final_mean = mean / len(datas)
        final_std = std / len(datas)

            # Zero Center and normalization the inputs
        if not only_calc_mean_std:
            for idx in range(len(datas)):
                #print(idx)
                datas[idx]["input"] = np.subtract(datas[idx]["input"].astype(np.float32),final_mean).astype(np.uint8)
                datas[idx]["input"] = np.divide(datas[idx]["input"].astype(np.float32), final_std).astype(np.uint8)
            # ZCA (TODO)

        return datas, final_mean, final_std
    else:
        final_mean = mean
        final_std = std
        #for validation and test set
        for idx in range(len(datas)):
            #print(idx)
            datas[idx]["input"] = np.subtract(datas[idx]["input"].astype(np.float32),final_mean).astype(np.uint8)
            datas[idx]["input"] = np.divide(datas[idx]["input"].astype(np.float32), final_std).astype(np.uint8)

        return datas
def main():
    # Get the data.
    trains = joblib.load("../data/CIKM2017_train/train_Imp_3x3.pkl")
    #testa_set = joblib.load("../data/CIKM2017_testA/testA_Imp_3x3_del_height_no.4.pkl")
    #testa_x = []

    #for item in testa_set:
    #    testa_x.append(item["input"])

    #testa_x = np.asarray(testa_x, dtype=np.int16).transpose((0,1,3,4,2))
    train_x, train_y, train_class = sample(trains)
    '''
    for i in range(10):
        np.random.shuffle(data_set)
    valid_data_num = int(len(data_set) / 10) #get 10% data for validation
    for i in range(10):
        valid_set = data_set[i * valid_data_num : (i+1) * valid_data_num ]
        train_set = data_set[0: i*valid_data_num]
        train_set.extend(data_set[(i+1)*valid_data_num:])
        train_out, train_mean, train_std = preprocessing(train_set, 0, 0, True )
        valid_out = preprocessing(valid_set, train_mean, train_std)

        testa_out = preprocessing(testa_set, train_mean, train_std)

        convert_to(train_out, "train_Imp_3x3_resample_normalization_"+str(i)+"_fold", is_test=False)
        convert_to(valid_out, "valid_Imp_3x3_resample_normalization_"+str(i)+"_fold", is_test=False)
        convert_to(testa_out, "testA_Imp_3x3_normalization_"+str(i)+"_fold", is_test=True)
    #joblib.dump(value=data_set, filename="../data/CIKM2017_train/train_Imp_3x3_classified_del_height_no.4.pkl",compress=3)
    '''
    h5fname = "../data/CIKM2017_train/train_Imp_3x3.h5"
    import h5py
    "write file"
    with h5py.File(h5fname, "w") as f:
        #f.create_dataset(name="testa_set_x", shape=testa_x.shape, data=testa_x, dtype=testa_x.dtype, compression="lzf", chunks=True)
        f.create_dataset(name="train_set_x", shape=train_x.shape, data=train_x, dtype=train_x.dtype, compression="lzf", chunks=True)
        f.create_dataset(name="train_set_y", shape=train_y.shape, data=train_y, dtype=train_y.dtype, compression="lzf", chunks=True)
        f.create_dataset(name="train_set_class", shape=train_class.shape, data=train_class, dtype=train_class.dtype, compression="lzf", chunks=True)

    return
if __name__ == "__main__":
    main()

