import sys
sys.path.append('./fnn')
import numpy as np
import pandas as pd
import scipy.io as scio
from myModel import LSTMEmbedding, MLPEmbedding, ETDEmbedding, AMIEmbedding, TICAEmbedding
from fnn.regularizers import FNN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.models import load_model
import pickle


def trainProcess():
    dd1 = scio.loadmat('./mydatasets/allSignal_rightTime.mat')
    x1 = dd1['xdata']
    y1 = dd1['yd'][0,:]
    y1 = y1/300
    random_dex = np.random.permutation(x1.shape[0])
    xdata = x1[random_dex, :]
    ydata = y1[random_dex]

    log_path = './tensorboard/lstmfnn'

    # LSTM
    for idx in range(6):
        lstm_model = LSTMEmbedding(64,
                             time_window=128,
                             logdir=log_path,
                             train_step = 1,
                             latent_regularizer=FNN(10),
                             random_state=0
                             )
        left = idx*128*3
        right = left+128*3
        temp = xdata[:,left:right]
        temp1 = temp.reshape((-1,128))  #确认reshape之后每128是不是计划中的
        rd_idx = np.random.permutation(temp1.shape[0])
        tdata = lstm_model.fit_transform(temp1[rd_idx,:])

        file_name = "./model/lstmfnn_encoder_" + str(idx) + ".h5"
        lstm_model.model.encoder.save(file_name)

    print('done')


def testProcess():
    with open('./mydatasets/fnn_de2.pickle','rb') as file:
        value = pickle.load(file)
        file.close()
    xdata = value['xdata'].reshape(-1,3,6,64)
    ydata = value['ydata'].reshape(-1,1)
    model = load_model('./h5model/fnn0804.h5')
    # ypred = model.predict(xdata,batch_size=1000).reshape(-1,1)
    # mape = np.mean(np.abs(ypred-ydata)/ydata)
    score = model.evaluate(xdata,ydata)
    print(score)

def generatePickle():
    dd1 = scio.loadmat('./mydatasets/allSignal_rightTime.mat')
    x1 = dd1['xdata']
    y1 = dd1['yd'][0, :]
    ydata = y1 / 300

    xdata = np.zeros((x1.shape[0],1))
    for idx in range(18):
        model_idx = int(idx/3)
        model_path = './model/encoder' + str(model_idx) + '.h5'
        model = load_model(model_path)

        left = idx * 128
        right = left + 128
        temp = x1[:, left:right]
        temp1 = temp.reshape((-1, 128))  # 确认reshape之后每128是不是计划中的
        rd_idx = np.random.permutation(temp1.shape[0])
        td = model.predict(temp1[rd_idx, :])
        xdata = np.hstack((xdata,td))

    value = {'xdata':xdata[:,1:],'ydata':ydata}
    file = open('./mydatasets/mlp.pickle','wb')
    pickle.dump(value,file)
    file.close()


generatePickle()
