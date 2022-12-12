import multilunarlander_v0
import numpy as np
import matplotlib.pyplot as plt
import keras
import random
import keras.backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from collections import deque
from tqdm import tqdm

############################################################################################

######################################### defining parameters ####################################################

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 128  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.12

# Environment settings
EPISODES = int(1e5)

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

############################################################################################

############################################## Class DQN agent ##############################################
# Agent class
class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        model = keras.Sequential()
        model.add(keras.Input(shape = 12))
        model.add(keras.layers.Dense(128,activation='relu'))
        model.add(keras.layers.Dense(64, activation='relu'))
        #model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(4, activation='softmax'))
        # compile the model with a mean squared error loss and an Adam optimizer
        #model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        #model.summary()
        return model

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

########################################################################################################

def initialize_memory(num_agents):

  memory_dict = {}
  for agent in num_agents:
      memory_dict[agent] = []

  return memory_dict


#############################################  main Loop  ###############################################

#setting up environment
env = multilunarlander_v0.env(n_landers=3, position_noise=1e-3, angle_noise=1e-3, terminate_reward=-100.0, damage_reward=-100.0, shared_reward=False,
terminate_on_damage=False, remove_on_damage=False, max_cycles= 500,successful_reward=200)

num_agents = env.possible_agents
observation_dim = env.observation_space(num_agents[0]).shape
action_dim = env.action_space(num_agents[0]).n
print(f"Agents : {num_agents}\n  Observation : {observation_dim}\n Action : {action_dim}")

replay_state_memory = initialize_memory(env.possible_agents)

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
        if len(replay_state_memory[agents]) > 0:
          agent.update_replay_memory((replay_state_memory[agents][-1], action, rewards, new_state, termination))
          agent.train(termination, step)

        replay_state_memory[agents].append(new_state)
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