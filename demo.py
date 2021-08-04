import numpy as np
import matplotlib.pyplot as plt

## Import from local directory
import sys
sys.path.insert(0, './fnn')
from fnn.models import LSTMEmbedding, MLPEmbedding
from fnn.regularizers import FNN

plt.rcParams['lines.linewidth'] = .1
plt.rcParams['axes.prop_cycle'] = plt.cycler(color="k")

# sol = np.loadtxt('datasets/torus.csv.gz', delimiter=',')
# obs = sol[0]
#
# plt.figure()
# plt.plot(sol[1][:10000], sol[2][:10000])
#
# plt.figure()
# plt.plot(obs[:1000], linewidth=1)
#
# model = MLPEmbedding(10, time_window=20, latent_regularizer=FNN(1e-1), random_state=0)
# coords = model.fit_transform(obs[:5000])
#
# plt.figure()
# plt.semilogy(model.train_history.history["loss"], linewidth=2)
#
#
# plt.figure()
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,4))
# ax[0].plot(coords[:, 0], coords[:, 1])
# ax[1].plot(coords[:, 1], coords[:, 2])
# ax[2].plot(coords[:, 0], coords[:, 2])




# multi-variate: Load dataset and select observation
sol = np.loadtxt('datasets/pendulum_train.csv.gz', delimiter=',')
obs = sol[:2, :].T

# Construct model, note the n_features argument
model = MLPEmbedding(10,time_window=10,latent_regularizer=FNN(1.0),random_state=0,n_features=2)
# Fit and apply embedding
coords = model.fit_transform(obs, learning_rate=1e-2)

# Plot embedding
# plt.figure()
# plt.semilogy(model.train_history.history["loss"], linewidth=2)
#
# plt.figure()
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,4))
# ax[0].plot(coords[:, 0], coords[:, 1])
# ax[1].plot(coords[:, 1], coords[:, 2])
# ax[2].plot(coords[:, 0], coords[:, 2])

from mpl_toolkits.mplot3d import axes3d
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection= '3d')
ax.plot(coords[:,0], coords[:,1], coords[:,2], color='k')
plt.show()
plt.legend()