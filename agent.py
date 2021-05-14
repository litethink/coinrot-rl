#================================================================ 
#   Idea by      : PyLessons
#   Created date: 2021-05-14
#   Description : Trading Crypto with Reinforcement Learning 
#================================================================

import copy
import json
import numpy as np
import os
from datetime import datetime
from tensorboardX import SummaryWriter
from tensorflow.keras.optimizers import Adam
from model import ActorModel,CriticModel

import pdb
def d():
    pdb.set_trace()


        

class BaseAgent:
    """docstring for BaseAgent"""
    def __init__(self):
        self.optimizer = Adam
        self.action_space = np.array([0, 1, 2])
        self.state_size = (self.lookback_window_size, 5 + self.depth) # 5 standard OHCL information + market and indicators
        # Create shared Actor-Critic network model
        self.Actor = ActorModel(input_shape=self.state_size, action_space = self.action_space.shape[0], lr=self.lr, 
            optimizer = self.optimizer, model=self.model)
        self.Critic = CriticModel(input_shape=self.state_size, action_space = self.action_space.shape[0], lr=self.lr, 
            optimizer = self.optimizer, model=self.model)

    def act(self, state):
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.predict(np.expand_dims(state, axis=0))[0]
        action = np.random.choice(self.action_space, p=prediction)
        return action, prediction

class TrainAgent(BaseAgent):
    # A custom Bitcoin trading agent
    def __init__(self,lookback_window_size,depth,**kwargs):
        self.lookback_window_size = lookback_window_size
        self.depth = depth
        self.model = kwargs.get("model","CNN")
        self.comment = kwargs.get("comment","")
        self.lr = kwargs.get("lr",0.00005)
        self.epochs = kwargs.get("epochs",1)
        self.batch_size = kwargs.get("batch_size",32)
        super(TrainAgent, self).__init__()

        # Action space from 0 to 3, 0 is hold, 1 is buy, 2 is sell
        # folder to save models
        self.log_name = '{}_{}'.format(datetime.now().strftime("%Y%m%d%H%M"),"crypto_trader")
        # State size contains Market+Orders+Indicators history for the last lookback_window_size steps
        #for save train data 
        self.train_path = "{}/{}".format("train",self.log_name)
        if not os.path.exists("train"):
            os.makedirs("train")
        os.makedirs(self.train_path)

    def create_writer(self, initial_balance, normalize_value, train_episodes):
        self.replay_count = 0
        self.writer = SummaryWriter('{}/{}'.format("runs",self.log_name))

        # Create folder to save models
        self.start_training_log(initial_balance, normalize_value, train_episodes)
            
    def start_training_log(self, initial_balance, normalize_value, train_episodes):      
        # save training parameters to Parameters.json file for future
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        params = {
            "training start": current_date,
            "initial balance": initial_balance,
            "training episodes": train_episodes,
            "lookback window size": self.lookback_window_size,
            "depth": self.depth,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch size": self.batch_size,
            "normalize value": normalize_value,
            "model": self.model,
            "comment": self.comment,
            "saving time": "",
            "actor name": "",
            "critic name": "",
        }

        with open("{}/{}".format(self.train_path,"parameters.json"), "w") as write_file:
            json.dump(params, write_file, indent=4)


    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.95, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, predictions, dones, next_states):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Get Critic network predictions 
        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)
        
        # Compute advantages
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
        '''
        plt.plot(target,'-')
        plt.plot(advantages,'.')
        ax=plt.gca()
        ax.grid(True)
        plt.show()
        '''
        # stack everything to numpy array
        y_true = np.hstack([advantages, predictions, actions])
        # training Actor and Critic networks
        a_loss = self.Actor.model.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=True, batch_size=self.batch_size)
        c_loss = self.Critic.model.fit(states, target, epochs=self.epochs, verbose=0, shuffle=True, batch_size=self.batch_size)

        self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
        self.replay_count += 1

        return np.sum(a_loss.history['loss']), np.sum(c_loss.history['loss'])
        
    def save(self, name="crypto_trader", score="", args=[]):
        # save keras model weights
        actor_path = "{}/{}_{}_actor.h5".format(self.train_path,score,name)
        critic_path = "{}/{}_{}_critic.h5".format(self.train_path,score,name)
        self.Actor.model.save_weights(actor_path)
        self.Critic.model.save_weights(critic_path)

        # update json file settings
        if score != "":
            params_path = "{}/{}".format(self.train_path,"parameters.json")
            with open(params_path, "r") as json_file:
                params = json.load(json_file)
            params["saving time"] = datetime.now().strftime('%Y-%m-%d %H:%M')
            params["actor name"] = f"{score}_{name}_actor.h5"
            params["critic name"] = f"{score}_{name}_critic.h5"
            with open(params_path, "w") as write_file:
                json.dump(params, write_file, indent=4)

        # log saved model arguments to file
        if len(args) > 0:
            log_path = "{}/{}".format(self.train_path,"train.log")
            with open(log_path, "a+") as log:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                arguments = ""
                for arg in args:
                    arguments += f", {arg}"
                log.write(f"{current_time}{arguments}\n")


class TestAgent(BaseAgent):
    """docstring for TestAgent"""
    def __init__(self,**kwargs):
        self.lookback_window_size = kwargs.get("lookback window size")
        self.depth = kwargs.get("depth")
        self.lr = kwargs.get("lr")
        self.model = kwargs.get("model")
        super(TestAgent, self).__init__()        

    def load(self, folder):
        # load keras model weights
        self.Actor.model.load_weights(os.path.join(folder, "_crypto_trader_actor.h5"))
        self.Critic.model.load_weights(os.path.join(folder, "_crypto_trader_critic.h5"))
