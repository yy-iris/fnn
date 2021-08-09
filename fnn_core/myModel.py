"""
TensorFlow functions to support the false nearest neighbor regularizer
"""
import tensorflow as tf
# import tensorflow_addons as tfa
import numpy as np
import warnings
# from utils import standardize_ts, hankel_matrix, resample_dataset
from tica import tICA

# tf.__version__ must be greater than 2
# print(len(tf.config.list_physical_devices('GPU')), "GPUs available.")
# Suppress some warnings that appeared in tf 2.2
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

use_legacy = False  ## for development
if use_legacy:
    warnings.warn("Legacy mode is enabled, archival versions of functions will be used")

from sklearn.decomposition import PCA, SparsePCA, KernelPCA, FastICA

from networks import *
from regularizers import FNN


class TimeSeriesEmbedding:
    """Base class for time series embedding

    Properties
    ----------

    train_history : dict
        The training history of the model

    model : "lstm" | "mlp" | "tica" | "etd" | "delay"
        The type of model to use for the embedding.

    n_latent : int
        The embedding dimension

    n_features : int
        The number of channels in the time series

    **kwargs : dict
        Keyword arguments passed to the model

    """

    def __init__(
            self,
            n_latent,
            time_window=10,
            n_features=1,
            logdir = None,
            train_step = 1,
            random_state=None,
            **kwargs
    ):
        self.n_latent = n_latent
        self.time_window = time_window
        self.n_features = n_features
        self.random_state = random_state
        self.logdir = logdir
        self.train_step = train_step

    def fit(self, X, y=None):
        raise AttributeError("Derived class does not contain method.")

    def transform(self, X, y=None):
        raise AttributeError("Derived class does not contain method.")

    def fit_transform(self, X, y=None, **kwargs):
        """Fit the model with a time series X, and then embed X.

        Parameters
        ----------
        X : array-like, shape (n_timepoints, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : None
            Ignored variable.

        kwargs : keyword arguments passed to the model's fit() method

        Returns
        -------
        X_new : array-like, shape (n_timepoints, n_components)
            Transformed values.
        """
        self.fit(X, **kwargs)
        return self.transform(X)



from sklearn.metrics import mutual_info_score
from scipy.signal import savgol_filter, argrelextrema


class NeuralNetworkEmbedding(TimeSeriesEmbedding):
    """Base class autoencoder model for time series embedding

    Properties
    ----------

    n_latent : int
        The embedding dimension

    n_features : int
        The number of channels in the time series

    **kwargs : dict
        Keyword arguments passed to the model

    """

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        # # Default latent regularizer is FNN
        # if np.isscalar(latent_regularizer):
        #     latent_regularizer = FNN(latent_regularizer)

    def fit(
            self,
            X,
            y=None,
            subsample=None,
            tau=0,
            learning_rate=1e-3,
            batch_size=1000,
            loss='mse',
            verbose=0,
            optimizer="adam",
            early_stopping=False
    ):
        """Fit the model with a time series X

        Parameters
        ----------
        X : array-like, shape (n_timepoints, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : None
            Ignored variable.

        subsample : int or None
            If set to an integer, a random number of timepoints is selected
            equal to that integer

        tau : int
            The prediction time, or the number of timesteps to skip between
            the input and output time series


        Returns
        -------
        X_new : array-like, shape (n_timepoints, n_components)
            Transformed values.
        """

        # conver_to_tensor

        # if subsample:
        #     self.train_indices, _ = resample_dataset(
        #         X_train, subsample, random_state=self.random_state
        #     )
        #     X_train = X_train[self.train_indices]
        #     Y_train = Y_train[self.train_indices]

        optimizers = {
            "adam": tf.keras.optimizers.Adam(learning_rate=learning_rate),
            "nadam": tf.keras.optimizers.Nadam(learning_rate=learning_rate)
            # "radam": tfa.optimizers.RectifiedAdam(learning_rate=learning_rate),
        }

        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        self.model.compile(
            optimizer=optimizers[optimizer],
            loss=loss,
            # experimental_run_tf_function=False
        )

        # if early_stopping:
        #     callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=3)]
        # else:
        #     callbacks = [None]

        tensorboard = [tf.keras.callbacks.TensorBoard(log_dir=self.logdir),
                       tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=3)]
        self.train_history = self.model.fit(
            x=X,
            y=X,
            epochs=self.train_step,
            batch_size=batch_size,
            verbose=1,
            callbacks = tensorboard
        )

    def transform(self, X, y=None):
        # X_test = hankel_matrix(standardize_ts(X), self.time_window)
        # X_test = tf.expand_dims(tf.convert_to_tensor(X), axis=2)
        X_new = self.model.encoder.predict(X)
        return X_new

    # class CausalEmbedding(NeuralNetworkEmbedding):


#     """
#     Calculates strides and input size automatically
#     """
#     def __init__(
#         self,
#         *args,
#         network_shape=[10, 10],
#         **kwargs
#     ):
#         super().__init__(*args, **kwargs)
#         self.depth = len(network_shape)
#         print(self.n_latent)
#         stride_size = math.floor(math.log(self.time_window/self.n_latent, (2*self.depth)))
#         final_conv_size = math.ceil(self.time_window/(stride_size**(2*self.depth)))
#         time_window_new = final_conv_size*(stride_size**(2*self.depth))
#         #print(time_window_new, stride_size, final_conv_size)
#         if time_window_new != self.time_window:
#             self.time_window = time_window_new
#             print("Time window increased to ", str(time_window_new), ", an integer power",
#              "of stride size. If this is too large, decrease network depth or latent size")
#         print("Effective stride size is ", str(stride_size**2)) # each block does two downsampling
#         print("Final convolution size is ", str(final_conv_size))

#         self.model = CausalAutoencoder(
#             self.n_latent,
#             network_shape=network_shape,
#             conv_output_shape=final_conv_size,
#             strides = stride_size,
#             **kwargs
#         )


class LSTMEmbedding(NeuralNetworkEmbedding):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        kwargs.pop("time_window")
        if use_legacy:
            self.model = LSTMAutoencoderLegacy(
                self.n_latent,
                self.time_window,
                **kwargs
            )
        else:
            self.model = my_LSTMAutoencoder(
                n_latent=self.n_latent,
                time_window=self.time_window,
                **kwargs
            )

