import pandas as pd
import logging
import os
from collections import deque
import numpy as np
import tensorflow as tf
import random
import gym
from tqdm import tqdm
import matplotlib.pyplot as plt

BUFFER_UNBALANCE_GAP = 0.5


def actor_network(state_space, num_actions):
    inp = tf.keras.layers.Input(shape=state_space, dtype=tf.float32)

    outs = tf.keras.layers.Dense(100)(inp)
    outs = tf.keras.layers.BatchNormalization()(outs)
    outs = tf.keras.activations.relu(outs)

    outs = tf.keras.layers.Dense(100)(outs)
    actions = tf.keras.layers.Dense(num_actions, activation='tanh')(outs)

    model = tf.keras.Model(inp, actions)
    model.summary()
    return model


def critic_network(state_space, num_actions):
    inp = tf.keras.layers.Input(shape=state_space, dtype=tf.float32)
    inp_action = tf.keras.layers.Input(shape=num_actions, dtype=tf.float32)

    outs = tf.keras.layers.Concatenate(axis=1)([inp, inp_action])

    outs = tf.keras.layers.Dense(100)(outs)
    outs = tf.keras.layers.BatchNormalization()(outs)
    outs = tf.keras.activations.relu(outs)

    outs = tf.keras.layers.Dense(100)(outs)
    outputs = tf.keras.layers.Dense(1)(outs)

    # Outputs single value for give state-action
    model = tf.keras.Model([inp, inp_action], outputs)

    model.summary()
    return model


"""
Buffer system for the RL
"""


class ReplayBuffer:
    """
    Replay Buffer to store the experiences.
    """

    def __init__(self, buffer_size, batch_size):
        """
        Initialize the attributes.
        Args:
            buffer_size: The size of the buffer memory
            batch_size: The batch for each of the data request `get_batch`
        """
        self.buffer = deque(maxlen=int(buffer_size))  # with format of (s,a,r,s')

        # constant sizes to use
        self.batch_size = batch_size

        # temp variables
        self.p_indices = [BUFFER_UNBALANCE_GAP / 2]

    def append(self, state, action, r, sn, d):
        """
        Append to the Buffer
        Args:
            state: the state
            action: the action
            r: the reward
            sn: the next state
            d: done (whether one loop is done or not)
        """
        self.buffer.append([state, action, np.expand_dims(r, -1), sn, np.expand_dims(d, -1)])

    def get_batch(self, unbalance_p=True):
        """
        Get the batch randomly from the buffer
        Args:
            unbalance_p: If true, unbalance probability of taking the batch from buffer with
            recent event being more prioritized
        Returns:
            the resulting batch
        """
        # unbalance indices
        p_indices = None
        if random.random() < unbalance_p:
            self.p_indices.extend((np.arange(len(self.buffer) - len(self.p_indices)) + 1)
                                  * BUFFER_UNBALANCE_GAP + self.p_indices[-1])
            p_indices = self.p_indices / np.sum(self.p_indices)

        chosen_indices = np.random.choice(len(self.buffer),
                                          size=min(self.batch_size, len(self.buffer)),
                                          replace=False,
                                          p=p_indices)

        buffer = [self.buffer[chosen_index] for chosen_index in chosen_indices]

        return buffer

    def get_buffer_size(self):
        return len(self.buffer)


def update_target(model_target, model_ref, rho=0):
    """
    Update target's weights with the given model reference
    Args:
        model_target: the target model to be changed
        model_ref: the reference model
        rho: the ratio of the new and old weights
    """
    model_target.set_weights([rho * ref_weight + (1 - rho) * target_weight
                              for (target_weight, ref_weight) in
                              list(zip(model_target.get_weights(), model_ref.get_weights()))])


class Agent:
    """
    The Agent  that contains all the models
    """

    def __init__(self, state_space, action_space, gamma=0.9, rho=0.01,
                 std_noise=0.01, buffer_size=5000, batch_size=32, critic_lr=1e-2, actor_lr=1e-3):

        num_actions = action_space[0]
        self.actor_network = actor_network(state_space, num_actions)
        self.critic_network = critic_network(state_space, num_actions)
        self.actor_target_network = actor_network(state_space, num_actions)
        self.critic_target_network = critic_network(state_space, num_actions)

        # Making the weights equal initially
        self.actor_target_network.set_weights(self.actor_network.get_weights())
        self.critic_target_network.set_weights(self.critic_network.get_weights())

        self.buffer = ReplayBuffer(buffer_size, batch_size)
        self.gamma = tf.constant(gamma)
        self.rho = rho

        self.num_actions = num_actions
        self.std_noise = std_noise

        # optimizers
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr, amsgrad=True)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr, amsgrad=True)

        # define update weights with tf.function for improved performance
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(None, *state_space), dtype=tf.float32),
                tf.TensorSpec(shape=(None, num_actions), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, *state_space), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            ])
        def update_weights(s, a, r, sn, d):
            """
            Function to update weights with optimizer
            """
            with tf.GradientTape() as tape:
                # define target
                y = r + self.gamma * (1 - d) * self.critic_target_network([sn, self.actor_target_network(sn)])
                # because we dont have a terminal state
                # y = r + self.gamma * self.critic_target_network([sn, self.actor_target_network(sn)])
                # define the delta Q
                critic_loss = tf.math.reduce_mean(tf.math.abs(y - self.critic_network([s, a])))
            critic_grad = tape.gradient(critic_loss, self.critic_network.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic_network.trainable_variables))

            with tf.GradientTape() as tape:
                # define the delta mu
                actor_loss = -tf.math.reduce_mean(self.critic_network([s, self.actor_network(s)]))
            actor_grad = tape.gradient(actor_loss, self.actor_network.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor_network.trainable_variables))
            return critic_loss, actor_loss

        self.update_weights = update_weights

    def act(self, state, noise=True):
        """
        Run action by the actor network
        Args:
            state: the current state
            noise: whether noise is to be added to the result action (this improves exploration)
        Returns:
            the resulting action
        """
        state_ = state[np.newaxis, ...]
        action = self.actor_network(state_)[0].numpy()
        noise_action = np.random.randn(len(action)) * self.std_noise

        action = action + noise_action if noise else action
        return action

    def remember(self, state, action, reward, next_state, done):
        """
        Store states, reward, done value to the buffer
        """
        # record it in the buffer based on its reward
        self.buffer.append(state, action, reward, next_state, int(done))

    def learn(self):
        entry = self.buffer.get_batch(unbalance_p=False)
        """
        Run update for all networks (for training)
        """
        s, a, r, sn, d = zip(*entry)
        if self.buffer.get_buffer_size() < 100:
            return 0, 0
        c_l, a_l = self.update_weights(tf.convert_to_tensor(s, dtype=tf.float32),
                                       tf.convert_to_tensor(a, dtype=tf.float32),
                                       tf.convert_to_tensor(r, dtype=tf.float32),
                                       tf.convert_to_tensor(sn, dtype=tf.float32),
                                       tf.convert_to_tensor(d, dtype=tf.float32))

        update_target(self.actor_target_network, self.actor_network, self.rho)
        update_target(self.critic_target_network, self.critic_network, self.rho)

        return c_l, a_l

    def save_weights(self, path):
        """
        Save weights to `path`
        """
        parent_dir = os.path.dirname(path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        # Save the weights
        self.actor_network.save_weights(path + "an.h5")
        self.critic_network.save_weights(path + "cn.h5")
        self.critic_target_network.save_weights(path + "ct.h5")
        self.actor_target_network.save_weights(path + "at.h5")

    def load_weights(self, path):
        """
        Load weights from path
        """
        try:
            self.actor_network.load_weights(path + "an.h5")
            self.critic_network.load_weights(path + "cn.h5")
            self.critic_target_network.load_weights(path + "ct.h5")
            self.actor_target_network.load_weights(path + "at.h5")
        except OSError as err:
            logging.warning("Weights files cannot be found, %s", err)


if __name__ == "__main__":
    env = gym.make('Pendulum-v1')
    print(env.action_space)
    agent = Agent(env.observation_space.shape, env.action_space.shape)

    all_rewards = []
    for i_episode in tqdm(range(1000)):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0
        while not done:
            # env.render()
            action = agent.act(obs) * 2
            obs_, reward, done, info = env.step(action)

            agent.remember(obs, action, reward, obs_, done)
            agent.learn()

            obs = obs_
            ep_reward += reward
            ep_length += 1
        print("{}:{}".format(ep_length, ep_reward))
        all_rewards.append(ep_reward)
        agent.save_weights("model/")

    pd.DataFrame(all_rewards).plot()
    plt.show()

    # test the agent
    agent.load_weights("model/")

    for i in range(10):
        done = False
        obs = env.reset()
        while not done:
            env.render()
            action = agent.act(obs) * 2
            obs_, reward, done, info = env.step(action)
            obs = obs_

    env.close()
