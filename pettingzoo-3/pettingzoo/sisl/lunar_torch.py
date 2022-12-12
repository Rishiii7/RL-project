import multilunarlander_v0
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


############################################################################################

######################################### defining parameters ####################################################

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 10

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

############################################################################################

############################################## Class DQN agent ##############################################
class Net(nn.Module):
    def __init__(self) :
        super(Net,self).__init__()

        self.fc1 = nn.Linear(12,128)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(128,64)
        self.act2 = nn.ReLU()
        self.fc3  = nn.Linear(64,32)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(32,4)
        self.act4 = nn.Softmax()
        
    def forward(self, x):

        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.fc4(x)
        x = self.act4(x)

        return x
        



# Agent class
class DQNAgent:
    def __init__(self):
        super(DQNAgent, self).__init__()

        # Main model
        self.model = Net()

        # Target network
        self.target_model = Net()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
        

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)
        #print(f"Current state : {current_states.shape} Current Q list : {current_qs_list}")

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        #print(np.array(X).shape)
        #print(np.array(y).shape)
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

############################################################################################

#############################################  main Loop  ###############################################

#setting up environment
env = multilunarlander_v0.env(n_landers=3, position_noise=1e-3, angle_noise=1e-3, terminate_reward=-100.0, damage_reward=-100.0, shared_reward=False,
terminate_on_damage=False, remove_on_damage=False, max_cycles= 500,successful_reward=200)

num_agents = env.possible_agents
observation_dim = env.observation_space(num_agents[0]).shape
action_dim = env.action_space(num_agents[0]).n
print(f"Agents : {num_agents}\n  Observation : {observation_dim}\n Action : {action_dim}")


#Creating DQN agent
agent = DQNAgent()

#storing reward for each episode
ep_rewards = []

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    if current_state == None :
      current_state = np.zeros((12))

    for agents in env.agent_iter():

        #call last method
        new_state, rewards, termination, truncation, info = env.last()

        #print(f"New state : {new_state.shape} Rewards : {rewards}")

        # Transform new continous state to new discrete state and count reward
        episode_reward += rewards

        # This part stays mostly the same, the change is to query a model for Q values
        if termination or truncation:
            action = None
        
        elif isinstance(new_state, dict) and "action_mask" in new_state:
            action = np.random.choice(np.flatnonzero(new_state["action_mask"]))
        
        elif np.random.random() > epsilon:
            # Get action from Q table
            #print("\n into qs function")
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            #print("action taken randomly")
            action = np.random.randint(0, action_dim)

        #call step function
        env.step(action)

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        #print(f"Current state :{current_state.shape} , new_state : {new_state.shape}")
        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, rewards, new_state, termination))
        agent.train(termination, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)

    print(f"{episode} Finished !!")
    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
    
    

############################################################################################

"""
env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    print(observation.shape)
    print(reward)
    print(termination)
    print("================")
    # Sample random action for current agent
    action = None if termination or truncation else env.action_space(agent).sample()
    env.step(action)

"""