from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import joblib


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def convert_to(data_set, name):
    num_examples = len(data_set)

    filename = os.path.join("/mnt/guankai/CIKM/data/CIKM2017_train/", name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = data_set[index]["input"].tostring()
        label_raw = data_set[index]["label"].item()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _float_feature(label_raw),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


def main():
    # Get the data.
    data_set = joblib.load("/mnt/guankai/CIKM/data/CIKM2017_train/train_Imp_3x3_resampled.pkl")
    for i in range(10):
        data_set = np.random.permutation(data_set)
    valid_data_num = int(len(data_set) / 10) #get 10% data for validation
    valid_set = data_set[0 : valid_data_num ]
    train_set = data_set[valid_data_num  : ]
    convert_to(train_set, "train_Imp_3x3_resampled")
    convert_to(valid_set, "valid_Imp_3x3_resampled")
    return


if __name__ == '__main__':
    main()
