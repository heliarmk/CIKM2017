# coding: utf-8
import numpy as np
import os
from sklearn import preprocessing
from scipy import misc
import cv2
import warnings
import joblib
print('load data...')
trainData = joblib.load('../CIKM2017_train/train_Imp_3x3.pkl')
print('load over')
warnings.filterwarnings('ignore')
min_max_scaler = preprocessing.MinMaxScaler((0,255))
baseFold = '../CIKM2017_train/trainImagesMat'
if not os.path.isdir(baseFold):
    os.mkdir(baseFold)
for k in range(len(trainData)):
#for k in range(1):
    print('for datas:'+str(k))
    datas = trainData[k]['input']
    dShape = datas.shape
    #dirName = baseFold + '/' + str(k)
    #if not os.path.isdir(dirName):
    #    os.mkdir(dirName)
    concatImg = np.zeros((dShape[1]*(3+dShape[2]*2) - 3,dShape[0]*(3 + dShape[3]*2)-3))
    #print('size:'+str(concatImg.shape))
    for i in range(dShape[0]):
        for j in range(dShape[1]):
            indice = datas[i][j]
            indice_1d = np.reshape(indice,-1)
            indice_minmax_1d = min_max_scaler.fit_transform(indice_1d)
            indice_minmax = np.reshape(indice_minmax_1d,(dShape[2],dShape[3]))
            indice_minmax.astype('int')
            indice_resize = misc.imresize(indice_minmax,(202,202),'bilinear')
            
            rowS = (dShape[2]*2+3)*j
            rowE = (dShape[2]*2+3)*(j+1)-3
            columnS = (dShape[3]*2+3)*i
            columnE = (dShape[3]*2+3)*(i+1)-3
            
            #print('i:'+str(i)+',j:'+str(j)+',rows:'+str(rowS)+':'+str(rowE)+',columns:'+str(columnS)+':'+str(columnE))
            concatImg[rowS:rowE,columnS:columnE] = indice_resize
            
    pngName = baseFold + '/' + str(k) + '.png'
    cv2.imwrite(pngName,concatImg)
    
