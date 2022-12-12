import multilunarlander_v0
import numpy as np
from keras.models import save_model, load_model
from tqdm import tqdm


EPISODES = 10

env = pickle.load('finalized_model.pkl')
num_agents = env.possible_agents
observation_dim = env.observation_space(num_agents[0]).shape
action_dim = env.action_space(num_agents[0]).n
print(f"Agents : {num_agents}\n  Observation : {observation_dim}\n Action : {action_dim}")


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
        
        else :
            # Get action from Q table
            #print("\n into qs function")
            action = np.argmax(agents.get_qs(current_state))

        #call step function
        env.step(action)