#================================================================ 
#   Idea by      : PyLessons
#   Created date: 2021-05-14
#   Description : Trading Crypto with Reinforcement Learning 
#================================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D#, LSTM
from tensorflow.compat.v1.keras.layers import CuDNNLSTM as LSTM # only for GPU
from tensorflow.keras import backend as K
#tf.config.experimental_run_functions_eagerly(True) # used for debuging and development
import pdb
def d():
    pdb.set_trace()
tf.compat.v1.disable_eager_execution() # usually using this for fastest performance


gpus = tf.config.experimental.list_physical_devices('GPU')

if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: pass

class BaseModel:
    """docstring for A"""

    def init_X(self,model,X_input):
        if model.upper() == "CNN":
            X = Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh")(X_input)
            X = MaxPooling1D(pool_size=2)(X)
            X = Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")(X)
            X = MaxPooling1D(pool_size=2)(X)
            X = Flatten()(X)
            return X, None 
        elif model.upper() == "LSTM":           
            X = LSTM(512, return_sequences=True)(X_input)
            X = LSTM(256)(X)
            return X, None

        elif model.upper() == "DENSE":
            X = Flatten()(X_input)
            X = Dense(512, activation="relu")(X)
            return X, None
        else:
            #logger.error("")
            return None,"Model no implement."

    def init_model(self,X,X_input,lr,optimizer):
        raise NotImplementedError

    def predict(self,state):
        raise NotImplementedError


class ActorModel(BaseModel):

    def __init__(self, input_shape,action_space, lr, optimizer,model):
        self.model = None
        self.action_space = action_space
        X_input = Input(input_shape)
        X,err = self.init_X(model,X_input)
        self.init_model(X,X_input,lr,optimizer)

    def init_model(self,X,X_input,lr,optimizer):

        A = Dense(512, activation="relu")(X)
        A = Dense(256, activation="relu")(A)
        A = Dense(64, activation="relu")(A)
        output = Dense(self.action_space, activation="softmax")(A)
        self.model = Model(inputs = X_input, outputs = output)
        self.model.compile(loss=self._ppo_loss, optimizer=optimizer(lr=lr))
        return "Actor model implemented.",None

    def _ppo_loss(self, y_true, y_pred):
        # Defined in https://arxiv.org/abs/1707.06347
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001
        prob = actions * y_pred
        old_prob = actions * prediction_picks
        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)
        ratio = K.exp(K.log(prob) - K.log(old_prob))
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)  
        total_loss = actor_loss - entropy
        return total_loss

    def predict(self, state):
        return self.model.predict(state)

class CriticModel(BaseModel):

    def __init__(self, input_shape, action_space, lr, optimizer,model):
        self.model = None
        self.action_space = action_space
        X_input = Input(input_shape)
        X,err = self.init_X(model,X_input)
        self.init_model(X,X_input,lr,optimizer)

    def init_model(self,X,X_input,lr,optimizer):

        V = Dense(512, activation="relu")(X)
        V = Dense(256, activation="relu")(V)
        V = Dense(64, activation="relu")(V)
        value = Dense(1, activation=None)(V)
        self.model = Model(inputs=X_input, outputs = value)
        self.model.compile(loss=self._ppo2_loss, optimizer=optimizer(lr=lr))
        return "Critic model implemented.",None

    def _ppo2_loss(self, y_true, y_pred):
        value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
        return value_loss

    def predict(self, state):
        return self.model.predict([state, np.zeros((state.shape[0], 1))])
