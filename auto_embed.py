
import numpy as np
import pandas as pd
import scipy.io as scio
import autokeras as ak
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# dd1 = scio.loadmat('./mydatasets/allSignal_rightTime.mat')
# x1 = dd1['xdata']
# y1 = dd1['yd'][0, :]
# y1 = y1 / 300
# random_dex = np.random.permutation(x1.shape[0])
# xdata = x1[random_dex, :]
# ydata = y1[random_dex]
#
# left = 128*3
# right = left+128*3
# temp = xdata[:,left:right]
# temp1 = temp.reshape((-1,128,3))
#
# ## build the network
# signal_input = ak.Input()
# node1 = ak.RNNBlock(return_sequences=True)(signal_input)
# # node2 = ak.RNNBlock()(node1)
# output_node = ak.RegressionHead(loss='mean_absolute_percentage_error',metrics=['mean_absolute_error'])(node1)
# model = ak.AutoModel(inputs=[signal_input],outputs=[output_node],max_trials=2)
#
#
# model.fit(temp1, temp1, batch_size=500, epochs=2, validation_split=0.2)
# my_model = model.export_model()
# my_model.save("./h5model/onlyfmap_0723.h5")

xtrain = np.random.rand(200,32,10)
ytrain = np.random.randint(5,size=200)
ytrain = tf.keras.utils.to_categorical(ytrain)
input = ak.Input()
output = input
output = ak.RNNBlock()(output)
output = ak.ClassificationHead()(output)

model = ak.AutoModel(input,output)
model.fit(xtrain,ytrain, epochs=1,batch_size=100,verbose = False)
print('test')