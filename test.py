import sys
sys.path.append('./fnn_core')
import numpy as np
import pandas as pd
import scipy.io as scio
from fnn_core.myModel import LSTMEmbedding, MLPEmbedding, ETDEmbedding, AMIEmbedding, TICAEmbedding,ConvEmbedding
from fnn_core.regularizers import FNN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.models import load_model
import pickle
import autokeras as ak
from tensorflow.keras.callbacks import TensorBoard


def trainFnn_lstm():
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
                             n_features=3,
                             logdir=log_path,
                             train_step = 1,
                             latent_regularizer=FNN(10),
                             random_state=0
                             )
        left = idx*128*3
        right = left+128*3
        temp = xdata[:,left:right]
        if lstm_model.n_features==1:
            temp1 = temp.reshape((-1,128))  #确认reshape之后每128是不是计划中的
            temp1 = tf.expand_dims(tf.convert_to_tensor(temp1), axis=2)
        else:
            temp1 = temp.reshape((-1,lstm_model.n_features, 128))
            temp1 = tf.transpose(tf.convert_to_tensor(temp1),perm=[0,2,1])
        tdata = lstm_model.fit_transform(temp1)

        file_name = "./model/lstmfnn_train1_latent16/lstmfnn_encoder_" + str(idx) + ".h5"
        lstm_model.model.encoder.save(file_name)

    print('done')

def trainFnn_conv():
    dd1 = scio.loadmat('./mydatasets/allSignal_rightTime.mat')
    x1 = dd1['xdata']
    y1 = dd1['yd'][0, :]
    y1 = y1 / 300
    random_dex = np.random.permutation(x1.shape[0])
    xdata = x1[random_dex, :]
    ydata = y1[random_dex]

    log_path = './tensorboard/lstmfnn'

    # Conv
    n_feature = 3
    n_latent = 32
    time_window = 128
    for idx in range(6):
        model = ConvEmbedding(n_latent, time_window=time_window, n_features=n_feature,logdir=log_path,train_step=1,
                               latent_regularizer=FNN(10),
                               random_state=0
                               )
        left = idx * 128 * 3
        right = left + 128 * 3
        temp = xdata[:, left:right]
        temp1 = temp.reshape((-1, n_feature, time_window))
        temp1 = tf.expand_dims(tf.convert_to_tensor(temp1), axis=1)
        tdata = model.fit_transform(temp1)

        file_name = "./model/cnnfnn_train1/cnnfnn_encoder_" + str(idx) + ".h5"
        model.model.encoder.save(file_name)

    print('done')

def trainNetwork_conv(xdata, ydata, channel=1):
    # with open('./mydatasets/lstm_train1_ft3.pickle','rb') as file:
    #     value = pickle.load(file)
    #     file.close()
    # xdata = value['xdata']
    # ydata = value['ydata']
    random_dex = np.array(pd.read_csv('./random_dex.csv'))[:,1:]
    split = int(random_dex.shape[0]*0.7)
    train_idx = random_dex[:split]
    test_idx = random_dex[split:]
    xtrain = xdata[train_idx,:].reshape(-1,channel,6,64)
    ytrain = ydata[train_idx].reshape(-1,1)
    xtest = xdata[test_idx, :].reshape(-1, channel, 6, 64)
    ytest = ydata[test_idx].reshape(-1, 1)
    input = keras.layers.Input(shape=(channel, 6, 64))

    k1 = layers.Conv2D(64, kernel_size=(1,2), strides=(1,1), activation='relu', padding='same')(input)
    k2 = layers.Conv2D(32, kernel_size=(1,2), strides=(1,1), activation='relu', padding='same')(k1)
    f1 = layers.Flatten()(k2)
    k3 = layers.Dense(64, activation='relu')(f1)
    k4 = layers.Dense(32, activation='relu')(k3)
    output = layers.Dense(1)(k4)

    model = keras.models.Model(inputs=input, outputs=output)

    model.compile(optimizer=keras.optimizers.RMSprop(1e-3), loss='mean_absolute_percentage_error', metrics=['mean_absolute_error'])
    model.fit(xtrain, ytrain, batch_size=500, epochs=200, validation_data=(xtest,ytest))
    model.compile(optimizer=keras.optimizers.RMSprop(1e-4), loss='mean_absolute_percentage_error',metrics=['mean_absolute_error'])
    model.fit(xtrain, ytrain, batch_size=500, epochs=100, validation_data=(xtest,ytest))

    # model.save('./h5model/lstmfnn_train1_ft3.h5')
    # print('done')
    return model

def trainNetwork_dense():
    with open('./mydatasets/lstm_train200.pickle','rb') as file:
        value = pickle.load(file)
        file.close()
    xdata = value['xdata']
    ydata = value['ydata']
    random_dex = np.array(pd.read_csv('./random_dex.csv'))[:,1:]
    split = int(random_dex.shape[0]*0.7)
    train_idx = random_dex[:split]
    test_idx = random_dex[split:]
    xtrain = xdata[train_idx,:].reshape(-1,1152)
    ytrain = ydata[train_idx].reshape(-1,1)
    xtest = xdata[test_idx, :].reshape(-1,1152)
    ytest = ydata[test_idx].reshape(-1, 1)

    input = keras.layers.Input(shape=(3*6*64,))
    k3 = layers.Dense(256, activation='relu')(input)
    k4 = layers.Dense(32, activation='relu')(k3)
    output = layers.Dense(1)(k4)

    model = keras.models.Model(inputs=input, outputs=output)

    model.compile(optimizer=keras.optimizers.RMSprop(1e-3), loss='mean_absolute_percentage_error', metrics=['mean_absolute_error'])
    model.fit(xtrain, ytrain, batch_size=500, epochs=200, validation_data=(xtest,ytest))
    model.compile(optimizer=keras.optimizers.RMSprop(1e-4), loss='mean_absolute_percentage_error',metrics=['mean_absolute_error'])
    model.fit(xtrain, ytrain, batch_size=500, epochs=100, validation_data=(xtest,ytest))
    model.save('./h5model/mlpfnn_train200.h5')
    print('done')

def generatePickle(data_path, file_path,n_feature):
    dd1 = scio.loadmat(data_path)
    x1 = dd1['xdata']
    y1 = dd1['yd'][0, :]
    ydata = y1 / 300
    xdata = np.zeros((x1.shape[0],1))
    if n_feature ==1:
        for idx in range(18):
            model_idx = int(idx/3)
            model_path = file_path + '/lstmfnn_encoder_' + str(model_idx) + '.h5'
            model = load_model(model_path)

            left = idx * 128
            right = left + 128
            temp = x1[:, left:right]
            temp1 = temp.reshape((-1, 128))  # 确认reshape之后每128是不是计划中的
            td = model.predict(temp1)
            xdata = np.hstack((xdata,td))
    elif n_feature==3:
        for idx in range(6):
            model_idx = int(idx / 3)
            model_path = file_path + '/lstmfnn_encoder_' + str(model_idx) + '.h5'
            model = load_model(model_path)

            left = idx * 128*3
            right = left + 128*3
            temp = x1[:, left:right]
            temp1 = temp.reshape((-1,n_feature, 128))
            temp1 = tf.transpose(tf.convert_to_tensor(temp1), perm=[0, 2, 1])
            td = model.predict(temp1)
            xdata = np.hstack((xdata, td))

    # value = {'xdata':xdata[:,1:],'ydata':ydata}
    # file = open('./mydatasets/lstm_train1_ft3.pickle','wb')
    # pickle.dump(value,file)
    # file.close()
    return xdata[:, 1:], ydata

def testProcess():
    # step1: initial: determine the n_feature
    n_feature = 1
    if n_feature==3:
        channel = 1
    elif n_feature==1:
        channel = 3

    load_train_pickle = False
    load_test_pickle = False
    load_h5model = False
    encoder_path = './model/lstmfnn_train1'
    h5model_path = './h5model/lstm_train200.h5'
    train_mat_path = '../pronyTest/data/allSignal_rightTime.mat'
    test_mat_path = '../pronyTest/data/allSignal_rightTime_noise30.mat'
    train_pickle_path = './mydatasets/fnn_rightTime.pickle'
    test_pickle_path = './mydatasets/fnn_rightTime.pickle'

    if load_train_pickle:
        with open(train_pickle_path,'rb') as file:
            value = pickle.load(file)
            file.close()
        xdata = value['xdata']
        ydata = value['ydata']
        # value = {'xdata':xdata[:,1:],'ydata':ydata}
        # file = open('./mydatasets/lstm_train1_ft3.pickle','wb')
        # pickle.dump(value,file)
        # file.close()
    else:
        xdata, ydata = generatePickle(train_mat_path, encoder_path, n_feature)
        xdata = xdata.reshape(-1, channel, 6, 64)

    # step2: get xdata/ydata in rightTime signal based on encoder.h5, and train related network
    if load_h5model:
        model = load_model(h5model_path)
    else:
        model = trainNetwork_conv(xdata,ydata,channel=channel)

    #step3: evalue the network through niose/de2 data
    if load_test_pickle:
        with open(test_pickle_path,'rb') as file:
            value = pickle.load(file)
            file.close()
        xdata = value['xdata']
        ydata = value['ydata']
        # value = {'xdata':xdata[:,1:],'ydata':ydata}
        # file = open('./mydatasets/lstm_train1_ft3.pickle','wb')
        # pickle.dump(value,file)
        # file.close()
    else:
        xdata, ydata = generatePickle(test_mat_path, encoder_path, n_feature)
        xdata = xdata.reshape(-1, channel, 6, 64)

    score = model.evaluate(xdata,ydata)
    r2 = 1-score[1]/np.var(ydata)
    print(score)



def test_structure():
    input = keras.Input(shape=(1,3,128))
    x = layers.Conv2D(32,(1,4),(1,1),activation='relu',padding='same')(input)
    x = layers.MaxPool2D(pool_size=(1,2), strides=(1,1), padding='same')(x)
    x = layers.Conv2D(32,(1,2),(1,1),activation='relu',padding='same')(x)
    encoded = layers.MaxPool2D(pool_size=(1,2), strides=(1,1), padding='same')(x)

    input2 = keras.Input(shape=(1,3,32))
    x = layers.Conv2D(32,(1,2),(1,1),activation='relu',padding='same')(input2)
    x = layers.UpSampling2D(size=(1,1))(x)
    x = layers.Conv2D(128, (1, 4), (1, 1), activation='relu', padding='same')(x)
    x = layers.UpSampling2D(size=(1, 1))(x)
    output = layers.Conv2D(128, (1, 4), (1, 1), activation='relu', padding='same')(x)

    model1 = keras.Model(input,encoded)
    model2 = keras.Model(input2,output)
    # print(model1.summary())
    print(model2.summary())

testProcess()
