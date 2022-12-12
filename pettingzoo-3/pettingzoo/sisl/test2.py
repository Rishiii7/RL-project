import multiwalker_v9

env = multiwalker_v9.env(n_walkers=3, position_noise=1e-3, angle_noise=1e-3, forward_reward=1.0, terminate_reward=-100.0, fall_reward=-10.0, shared_reward=True,
terminate_on_fall=True, remove_on_fall=True, terrain_length=200, max_cycles=500)

env.reset()
for agent in env.agent_iter():
	observation, reward, termination, truncation, info = env.last()
	# Sample a random action for this agent
	action = None if termination or truncation else env.action_space(agent).sample()
	env.step(action)