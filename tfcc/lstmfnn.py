import sys
sys.path.append('../fnn_core')
import numpy as np
import pandas as pd
import scipy.io as scio
from myModel import LSTMEmbedding, MLPEmbedding, ETDEmbedding, AMIEmbedding, TICAEmbedding
from regularizers import FNN
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import axes3d
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.models import load_model
import pickle
import argparse
import os
import time

def trainFnn(epoch):
    # parser = argparse.ArgumentParser()
    # parser.description = 'please enter parameters ...'
    # parser.add_argument("-e", "--epoch", type = int, default="1")
    # args = parser.parse_args()

    log_path = '../tensorboard/lstmfnn'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    model_path = '../model/lstmfnn/epoch' + str(epoch) + '/'+str(int(time.time()))
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    dd1 = scio.loadmat('../mydatasets/allSignal_rightTime.mat')
    x1 = dd1['xdata']
    y1 = dd1['yd'][0,:]
    y1 = y1/300
    random_dex = np.random.permutation(x1.shape[0])
    xdata = x1[random_dex, :]
    ydata = y1[random_dex]

    # LSTM
    for idx in range(6):
        lstm_model = LSTMEmbedding(64,
                             time_window=128,
                             logdir=log_path,
                             train_step = epoch,
                             latent_regularizer=FNN(10),
                             random_state=0
                             )
        left = idx*128*3
        right = left+128*3
        temp = xdata[:,left:right]
        temp1 = temp.reshape((-1,128))  #确认reshape之后每128是不是计划中的
        rd_idx = np.random.permutation(temp1.shape[0])
        tdata = lstm_model.fit_transform(temp1[rd_idx,:])

        file_name = model_path + "/lstmfnn_encoder_" + str(idx) + ".h5"
        lstm_model.model.encoder.save(file_name)

    print('done')


trainFnn(10)
trainFnn(50)
trainFnn(100)
