# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 21:05:26 2017

@author: Cicely
"""
import numpy as np

import joblib


filename = "train.pkl"
train = joblib.load(filename)
kernelSize = 3
kernel = np.ones([kernelSize,kernelSize])/(kernelSize*kernelSize)


def myImputer(kernel,sample):
    kernelSize = kernel.shape[0];
    vectorSize = sample.shape[3];

    for fIndex in range(0,int(sample.shape[0])):
        for sIndex in range(0,int(sample.shape[1])):
            sslice = sample[fIndex][sIndex]
            #find the index of -1
            [x,y] = np.where(sslice == -1)
            #change -1 to 0
            sslice[sslice == -1] = 0
            if x.size == 0:
                continue
            #broaden the vector 
            tempVectorH = np.zeros([int((kernelSize-1)/2),vectorSize])
            tempVectorV = np.zeros([vectorSize-1+kernelSize,int((kernelSize-1)/2)])
            tempSlice = np.vstack((tempVectorH,sslice,tempVectorH))
            tempSlice = np.hstack((tempVectorV,tempSlice,tempVectorV))
            
            zeroSlice = np.zeros(sslice.shape);
            for k in range(len(x)):
                subSlice = tempSlice[x[k]:x[k]+kernelSize,y[k]:y[k]+kernelSize]
                imputerValue = np.sum(subSlice*kernel)
                zeroSlice[x[k],y[k]] = np.around(imputerValue)
            sslice += zeroSlice.astype("int32")


for sampleIndex in range(len(train)):
    sample = train[sampleIndex]['input']

    myImputer(kernel,sample)

joblib.dump(train, filename="train_Imp_3x3.pkl",compress=3)
