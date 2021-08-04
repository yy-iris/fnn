import numpy as np

## Import from local directory
import sys
sys.path.insert(0, './fnn')
from fnn.models import LSTMEmbedding, MLPEmbedding, ETDEmbedding, AMIEmbedding, TICAEmbedding
from fnn.regularizers import FNN

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
plt.rcParams['lines.linewidth'] = .02
plt.rcParams['axes.prop_cycle'] = plt.cycler(color="k")

sol = np.loadtxt('datasets/lorenz.csv.gz', delimiter=',')
sol = sol[:, ::10]
obs = sol[0, :5000] # only first coordinate is observable

plt.plot(sol[0, -1000:], linewidth=1)

# LSTM
lstm_model = LSTMEmbedding(10,
                     time_window=10,
                     latent_regularizer=FNN(10),
                     random_state=0
                     )
coords_lstm = lstm_model.fit_transform(obs)
print("LSTM complete")

# MLP
mlp_model = MLPEmbedding(10,
                     time_window=10,
                     latent_regularizer=FNN(10),
                     random_state=0
                     )
coords_mlp = mlp_model.fit_transform(obs, learning_rate=2e-4)
print("MLP complete")

# ICA
ica_model = TICAEmbedding(10, time_window=10, time_lag=0)
coords_ica = ica_model.fit_transform(obs)
print("ICA complete")

# tICA
tica_model = TICAEmbedding(10, time_window=10, time_lag=10)
coords_tica = tica_model.fit_transform(obs)
print("tICA complete")

# EigenDelay
etd_model = ETDEmbedding(10, time_window=10)
coords_etd = etd_model.fit_transform(obs)
print("ETD complete")

# Average Mutual Information
ami_model = AMIEmbedding(10, lag_cutoff=15)
coords_ami = ami_model.fit_transform(obs)
print("AMI complete")

