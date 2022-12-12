import copy
import math

import Box2D
import numpy as np
import pygame
from Box2D.b2 import (
	circleShape,
	contactListener,
	edgeShape,
	fixtureDef,
	polygonShape,
	revoluteJointDef,
)
from gymnasium import spaces
from gymnasium.utils import seeding
from pygame import gfxdraw

from _utils import Agent

MAX_AGENTS = 5

FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY = 12.0

VIEWPORT_W = 600
VIEWPORT_H = 400

W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE

STEP = 14 / SCALE
LANDER_SEPERATION = 10

CATEGORY_MOON = 0x0001 # default value for fixtures
CATEGORY_LANDER = 0x0002
CATEGORY_LEG = 0x0004
CATEGORY_PARTICLE = 0x0008

MASK_MOON = CATEGORY_LANDER | CATEGORY_LEG | CATEGORY_PARTICLE | CATEGORY_MOON
MASK_LANDER = CATEGORY_LANDER | CATEGORY_LEG | CATEGORY_MOON
MASK_LEG = CATEGORY_LEG | CATEGORY_LANDER | CATEGORY_MOON
MASK_PARTICLE = CATEGORY_MOON

# COLLIDE IS TRUE IF (mask[i] & cat[j]) != 0) and ((cat[i] & mask[j]) != 0)

# lander is made of lander(body) and 2 legs

# need to change contact criteria

class ContactDetector(contactListener):
	def __init__(self, env):
		contactListener.__init__(self)
		self.env = env

	def BeginContact(self, contact):

		# If lander body gets in contact with anything, lander is damaged
		for i, lander in enumerate(self.env.landers):
			if lander.lander is not None:
				if lander.lander in [contact.fixtureA.body, contact.fixtureB.body]:
					self.env.damaged_landers[i] = True

		# if any leg collides with anything else except the moon, lander is damaged
		for i, lander in enumerate(self.env.landers):
			if lander.lander is not None:
				for leg in lander.legs:
					if leg == contact.fixtureA.body:
						if self.env.moon != contact.fixtureB.body:
							self.env.damaged_landers[i] = True
					if leg == contact.fixtureB.body:
						if self.env.moon != contact.fixtureA.body:
							self.env.damaged_landers[i] = True

        # if any leg is in contact with the moon, make ground contact true
		for lander in self.env.landers:
			if lander.lander is not None:
				for leg in lander.legs:
					if leg == contact.fixtureA.body:
						if self.env.moon == contact.fixtureB.body:
							leg.ground_contact = True
					if leg == contact.fixtureB.body:
						if self.env.moon == contact.fixtureA.body:
							leg.ground_contact = True

	def EndContact(self, contact):
		# same as above, but remove ground contact
		for lander in self.env.landers:
			if lander.lander is not None:
				for leg in lander.legs:
					if leg == contact.fixtureA.body:
						if self.env.moon == contact.fixtureB.body:
							leg.ground_contact = False
					if leg == contact.fixtureB.body:
						if self.env.moon == contact.fixtureA.body:
							leg.ground_contact = False


class LunarLander(Agent):
	def __init__(
		self,
		world,
		init_x,
		init_y,
		seed=None,
		gravity: float = -10.0,
		enable_wind: bool = False,
		wind_power: float = 15.0,
		turbulence_power: float = 1.5,
		continuous: bool = False
	):
		self.world = world
		self.lander = None
		self.init_x = init_x
		self.init_y = init_y
		self.lander_id = -int(self.init_x)
		self._seed(seed)
		self.m_power = 0.0
		self.s_power = 0.0
		self.continuous = continuous

	def _destroy(self):
		if not self.lander:
			return
		self.world.DestroyBody(self.lander)
		self.lander = None
		self.world.DestroyBody(self.legs[0])
		self.world.DestroyBody(self.legs[1])
		self._clean_particles(True)
		self.m_power = 0.0
		self.s_power = 0.0

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]


	def _reset(self):
		self._destroy()
		init_x = self.init_x
		init_y = self.init_y
		self.lander: Box2D.b2Body = self.world.CreateDynamicBody(
			position=(init_x, init_y),
			angle=0.0,
			fixtures=fixtureDef(
				shape=polygonShape(
					vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
				),
				density=5.0,
				friction=0.1,
				groupIndex=self.lander_id,
				categoryBits=CATEGORY_LANDER,
				maskBits=MASK_LANDER,
				restitution=0.0,
			),  # 0.99 bouncy
		)
		self.lander.color1 = (128, 102, 230)
		self.lander.color2 = (77, 77, 128)
		self.lander.ApplyForceToCenter(
			(
				self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
				self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
			),
			True,
		)

		self.particles = []

		self.legs = []
		for i in [-1, +1]:
			leg = self.world.CreateDynamicBody(
				position=(init_x - i * LEG_AWAY / SCALE, init_y),
				angle=(i * 0.05),
				fixtures=fixtureDef(
					shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
					density=1.0,
					restitution=0.0,
					groupIndex=self.lander_id,
					categoryBits=CATEGORY_LEG,
					maskBits=MASK_LEG,
				),  
			)
			leg.ground_contact = False
			leg.color1 = (128, 102, 230)
			leg.color2 = (77, 77, 128)
			rjd = revoluteJointDef(
				bodyA=self.lander,
				bodyB=leg,
				localAnchorA=(0, 0),
				localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
				enableMotor=True,
				enableLimit=True,
				maxMotorTorque=LEG_SPRING_TORQUE,
				motorSpeed=+0.3 * i,  # low enough not to jump back into the sky
			)
			if i == -1:
				rjd.lowerAngle = (
					+0.9 - 0.5
				)  # The most esoteric numbers here, angled legs have freedom to travel within
				rjd.upperAngle = +0.9
			else:
				rjd.lowerAngle = -0.9
				rjd.upperAngle = -0.9 + 0.5
			leg.joint = self.world.CreateJoint(rjd)
			self.legs.append(leg)

		self.drawlist = [self.lander] + self.legs


	def _create_particle(self, mass, x, y, ttl):
		p = self.world.CreateDynamicBody(
			position=(x, y),
			angle=0.0,
			fixtures=fixtureDef(
				shape=circleShape(radius=2 / SCALE, pos=(0, 0)),
				density=mass,
				friction=0.1,
				categoryBits=CATEGORY_PARTICLE,
				maskBits=MASK_PARTICLE,
				restitution=0.3,
			),
		)
		p.ttl = ttl
		self.particles.append(p)
		self._clean_particles(False)
		return p

	def _clean_particles(self, all):
		while self.particles and (all or self.particles[0].ttl < 0):
			self.world.DestroyBody(self.particles.pop(0))

	def apply_action(self, action):

		if self.continuous:
			action = np.clip(action, -1, +1).astype(np.float32)
		#else:
			# Removing this for now to get past an error
			#assert self.action_space.contains(
			#	action
			#), f"{action!r} ({type(action)}) invalid "

		# Engines
		tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
		side = (-tip[1], tip[0])
		dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

		m_power = 0.0
		if (self.continuous and action[0] > 0.0) or (
			not self.continuous and action == 2
		):
			# Main engine
			if self.continuous:
				m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
				assert m_power >= 0.5 and m_power <= 1.0
			else:
				m_power = 1.0
			# 4 is move a bit downwards, +-2 for randomness
			ox = tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
			oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
			impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
			p = self._create_particle(
				3.5,  # 3.5 is here to make particle speed adequate
				impulse_pos[0],
				impulse_pos[1],
				m_power,
			)  # particles are just a decoration
			p.ApplyLinearImpulse(
				(ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power),
				impulse_pos,
				True,
			)
			self.lander.ApplyLinearImpulse(
				(-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
				impulse_pos,
				True,
			)

		s_power = 0.0
		if (self.continuous and np.abs(action[1]) > 0.5) or (
			not self.continuous and action in [1, 3]
		):
			# Orientation engines
			if self.continuous:
				direction = np.sign(action[1])
				s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
				assert s_power >= 0.5 and s_power <= 1.0
			else:
				direction = action - 2
				s_power = 1.0
			ox = tip[0] * dispersion[0] + side[0] * (
				3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
			)
			oy = -tip[1] * dispersion[0] - side[1] * (
				3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
			)
			impulse_pos = (
				self.lander.position[0] + ox - tip[0] * 17 / SCALE,
				self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
			)
			p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
			p.ApplyLinearImpulse(
				(ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power),
				impulse_pos,
				True,
			)
			self.lander.ApplyLinearImpulse(
				(-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
				impulse_pos,
				True,
			)

		self.m_power = m_power
		self.s_power = s_power

	# H/4 hardcoded as of now in pos.y

	def get_observation(self):
		pos = self.lander.position
		vel = self.lander.linearVelocity

		state = [
			(pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
			(pos.y - ((H / 4) + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
			vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
			vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
			self.lander.angle,
			20.0 * self.lander.angularVelocity / FPS,
			1.0 if self.legs[0].ground_contact else 0.0,
			1.0 if self.legs[1].ground_contact else 0.0,
		]
		assert len(state) == 8

		return state

	# these are very bounded in gym lander. change later

	@property
	def observation_space(self):
		# 8 original obs (pos, etc), 4 obs for positions of 2 neighboring landers
		return spaces.Box(
			low=np.float32(-np.inf),
			high=np.float32(np.inf),
			shape=(8 + 4,),
			dtype=np.float32,
		)

	@property
	def action_space(self):
		if self.continuous:
			# Action is two floats [main engine, left-right engines].
			# Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
			# Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
			return spaces.Box(-1, +1, (2,), dtype=np.float32)
		else:
			# Nop, fire left engine, main engine, right engine
			return spaces.Discrete(4)


class MultiLunarLanderEnv:

	metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

	def __init__(
		self,
		n_landers=3,
		position_noise=1e-3,
		angle_noise=1e-3,
		terminate_reward=-100.0,
		successful_reward=+100.0,
		damage_reward=-100.0,
		shared_reward=True,
		terminate_on_damage=True,
		remove_on_damage=True,
		max_cycles=500,
		render_mode=None,
		continuous: bool = False,
		gravity: float = -10.0,
		enable_wind: bool = False,
		wind_power: float = 15.0,
		turbulence_power: float = 1.5,
	):
		"""Initializes the `MultilanderEnv` class.
		n_landers: number of bipedal landers in environment
		position_noise: noise applied to agent positional sensor observations
		angle_noise: noise applied to agent rotational sensor observations
		forward_reward: reward applied for an agent standing, scaled by agent's x coordinate
		damage_reward: reward applied when an agent damages down
		shared_reward: whether reward is distributed among all agents or allocated locally
		terminate_reward: reward applied for each damageen lander in environment
		terminate_on_damage: toggles whether agent is done if it damages down
		terrain_length: length of terrain in number of steps
		max_cycles: after max_cycles steps all agents will return done
		"""

		assert (
			-12.0 < gravity and gravity < 0.0
		), f"gravity (current value: {gravity}) must be between -12 and 0"
		self.gravity = gravity

		if 0.0 > wind_power or wind_power > 20.0:
			warnings.warn(
				colorize(
					f"WARN: wind_power value is recommended to be between 0.0 and 20.0, (current value: {wind_power})",
					"yellow",
				),
			)
		self.wind_power = wind_power

		if 0.0 > turbulence_power or turbulence_power > 2.0:
			warnings.warn(
				colorize(
					f"WARN: turbulence_power value is recommended to be between 0.0 and 2.0, (current value: {turbulence_power})",
					"yellow",
				),
			)
		self.turbulence_power = turbulence_power
		self.continuous = continuous
		self.n_landers = n_landers
		self.position_noise = position_noise
		self.angle_noise = angle_noise
		self.damage_reward = damage_reward
		self.terminate_reward = terminate_reward
		self.successful_reward = successful_reward
		self.terminate_on_damage = terminate_on_damage
		self.local_ratio = 1 - shared_reward
		self.remove_on_damage = remove_on_damage
		self.seed_val = None
		self.seed()
		self.setup()
		self.screen = None
		self.isopen = True
		self.agent_list = list(range(self.n_landers))
		self.last_rewards = [0 for _ in range(self.n_landers)]
		self.last_dones = [False for _ in range(self.n_landers)]
		self.last_obs = [None for _ in range(self.n_landers)]
		self.max_cycles = max_cycles
		self.render_mode = render_mode
		self.frames = 0
		self.clock = None
		self.enable_wind = enable_wind
		self.wind_idx = np.random.randint(-9999, 9999)
		self.torque_idx = np.random.randint(-9999, 9999)


	def get_param_values(self):
		return self.__dict__


	def setup(self):

		self.viewer = None

		self.world = Box2D.b2World(gravity=(0, self.gravity))
		self.moon = None

		Gap = VIEWPORT_W / SCALE / (self.n_landers + 1)

		init_x = Gap
		init_y = VIEWPORT_H / SCALE
		self.start_x = [
			init_x + (Gap * i) for i in range(self.n_landers)
		]
		self.landers = [
			LunarLander(self.world, init_x=sx, init_y=init_y, seed=self.seed_val, continuous = self.continuous)
			for sx in self.start_x
		]
		self.num_agents = len(self.landers)
		self.observation_space = [agent.observation_space for agent in self.landers]
		self.action_space = [agent.action_space for agent in self.landers]

		self.total_agents = self.n_landers

		self.prev_shaping = np.zeros(self.n_landers)

	@property
	def agents(self):
		return self.landers

	def seed(self, seed=None):
		self.np_random, seed_ = seeding.np_random(seed)
		self.seed_val = seed_
		for lander in getattr(self, "landers", []):
			lander._seed(seed_)
		return [seed_]

	def _destroy(self):
		if not self.moon:
			return
		self.world.contactListener = None
		self.world.DestroyBody(self.moon)
		self.moon = None

		for lander in self.landers:
			lander._destroy()

	def close(self):
		if self.screen is not None:
			pygame.quit()
			self.isopen = False

	def reset(self):
		self.setup()
		self.world.contactListener_bug_workaround = ContactDetector(self)
		self.world.contactListener = self.world.contactListener_bug_workaround
		self.game_over = False
		self.damaged_landers = np.zeros(self.n_landers, dtype=bool) # crashed landers?
		self.successful_landers = np.zeros(self.n_landers, dtype=bool)
		self.prev_shaping = np.zeros(self.n_landers)
		self.scroll = 0.0  # del later?

		self._generate_moon()

		# self.drawlist = copy.copy(self.moon)
		self.drawlist = []

		for lander in self.landers:
			lander._reset()
			self.drawlist += lander.legs
			self.drawlist += [lander.lander]
		r, d, o = self.scroll_subroutine()
		self.last_rewards = [0 for _ in range(self.n_landers)]
		self.last_dones = [False for _ in range(self.n_landers)]
		self.last_obs = o
		self.frames = 0

		return self.observe(0)

	def scroll_subroutine(self):

		xpos = np.zeros(self.n_landers) # remove later
		obs = []
		done = False
		rewards = np.zeros(self.n_landers)

		for i in range(self.n_landers):
			if self.landers[i].lander is None:
				obs.append(np.zeros_like(self.observation_space[i].low))
				continue
			pos = self.landers[i].lander.position
			x, y = pos.x, pos.y
			xpos[i] = x

			lander_obs = self.landers[i].get_observation()

			class Pair:
				def __init__(self, num, x, y, dist):
					self.num = num
					self.x = x
					self.y = y
					self.dist = dist

			# Add two nearest neighbour positions
			#(pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
			#(pos.y - ((H / 4) + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
			neighbor_obs = []
			full_list = [] # has neighbour num, xpos, ypos, dist to current lander
			# get the two nearest neighbours
			for j in range(self.n_landers):
				if ((j == i) or self.successful_landers[j] or self.landers[j].lander == None):
					continue
				jpos = self.landers[j].lander.position
				xj, yj = jpos.x, jpos.y
				dist = ((x - xj)*(x - xj)) + ((y - yj)*(y - yj))
				full_list.append(Pair(j, xj, yj, dist))

			if full_list:
				full_list.sort(key=lambda x: x.dist, reverse=False)

			for neighbor in full_list:
				neighbor_obs.append((x - neighbor.x)/(VIEWPORT_W / SCALE / 2))
				neighbor_obs.append((y - neighbor.y)/(VIEWPORT_H / SCALE / 2))
				if len(neighbor_obs) == 4:
					break

			while len(neighbor_obs) != 4:
				neighbor_obs.append(0.0)

			obs.append(np.array(lander_obs + neighbor_obs))

			shaping = (
				- 100 * np.sqrt(lander_obs[0] * lander_obs[0] + lander_obs[1] * lander_obs[1]) # dist to zone
				- 100 * np.sqrt(lander_obs[2] * lander_obs[2] + lander_obs[3] * lander_obs[3]) # velocity 
				- 100 * abs(lander_obs[4]) # angle from horiz
				+ 10 * lander_obs[6] # left leg contact
				+ 10 * lander_obs[7] # right leg contact
			)  # And ten points for legs contact, the idea is if you
			# lose contact again after landing, you get negative reward

			if neighbor_obs[0] != 0.0:
				shaping += (30 * np.sqrt(neighbor_obs[0] * neighbor_obs[0] + neighbor_obs[1] * neighbor_obs[1]))

			if neighbor_obs[2] != 0.0:
				shaping += (30 * np.sqrt(neighbor_obs[2] * neighbor_obs[2] + neighbor_obs[3] * neighbor_obs[3]))

			rewards[i] = shaping - self.prev_shaping[i]
			self.prev_shaping[i] = shaping

			rewards[i] -= (
				self.landers[i].m_power * 0.30
			)  # less fuel spent is better, about -30 for heuristic landing
			rewards[i] -= self.landers[i].s_power * 0.03

			# if lander goes out of bounds, damage it

			if abs(lander_obs[0]) >= 1.0:
				# negative reward is given below - same as damaged reward
				self.damaged_landers[i] = True

			if ((not self.landers[i].lander.awake) and (not self.damaged_landers[i])):
				# landed safely and is stable
				self.successful_landers[i] = True



		done = [False] * self.n_landers
		for i, (damaged, successful, lander) in enumerate(zip(self.damaged_landers, self.successful_landers, self.landers)):
			if damaged:
				rewards[i] += self.damage_reward
				if self.remove_on_damage:
					lander._destroy()
				if not self.terminate_on_damage:
					rewards[i] += self.terminate_reward
				done[i] = True
			if successful:
				rewards[i] += self.successful_reward
				# automatically remove on successful landing with +ve reward
				lander._destroy()
				done[i] = True

		# should i put this above the top for loop?
		if (
			(self.terminate_on_damage and np.sum(self.damaged_landers) > 0)
			# or self.game_over
		):
			rewards += self.terminate_reward
			done = [True] * self.n_landers

		return rewards, done, obs

	def step(self, action, agent_id, is_last):

		assert self.landers[agent_id].lander is not None, agent_id

		# Update wind
		if self.enable_wind and not (
			self.landers[agent_id].legs[0].ground_contact or self.landers[agent_id].legs[1].ground_contact
		):
			# the function used for wind is tanh(sin(2 k x) + sin(pi k x)),
			# which is proven to never be periodic, k = 0.01
			wind_mag = (
				math.tanh(
					math.sin(0.02 * self.wind_idx)
					+ (math.sin(math.pi * 0.01 * self.wind_idx))
				)
				* self.wind_power
			)
			self.wind_idx += 1
			self.landers[agent_id].ApplyForceToCenter(
				(wind_mag, 0.0),
				True,
			)

			# the function used for torque is tanh(sin(2 k x) + sin(pi k x)),
			# which is proven to never be periodic, k = 0.01
			torque_mag = math.tanh(
				math.sin(0.02 * self.torque_idx)
				+ (math.sin(math.pi * 0.01 * self.torque_idx))
			) * (self.turbulence_power)
			self.torque_idx += 1
			self.landers[agent_id].ApplyTorque(
				(torque_mag),
				True,
			)

		self.landers[agent_id].apply_action(action)

		if is_last:
			self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
			rewards, done, mod_obs = self.scroll_subroutine()
			self.last_obs = mod_obs
			global_reward = rewards.mean()
			local_reward = rewards * self.local_ratio
			self.last_rewards = (
				global_reward * (1.0 - self.local_ratio)
				+ local_reward * self.local_ratio
			)
			self.last_dones = done
			self.frames = self.frames + 1

		if self.render_mode == "human":
			self.render()

	def get_last_rewards(self):
		return dict(
			zip(
				list(range(self.n_landers)),
				map(lambda r: np.float64(r), self.last_rewards),
			)
		)

	def get_last_dones(self):
		return dict(zip(self.agent_list, self.last_dones))

	def get_last_obs(self):
		return dict(
			zip(
				list(range(self.n_landers)),
				[lander.get_observation() for lander in self.landers],
			)
		)

	def observe(self, agent):
		o = self.last_obs[agent]
		o = np.array(o, dtype=np.float32)
		return o

	def render(self, close=False):
		if close:
			self.close()
			return

		# offset = 200  # compensates for the negative coordinates
		# render_scale = SCALE / self.package_scale / 0.75
		if self.screen is None:
			pygame.init()
			self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))

		if self.clock is None:
			self.clock = pygame.time.Clock()

		self.surf = pygame.Surface(
			# (VIEWPORT_W + self.scroll * render_scale + offset, VIEWPORT_H)
			(VIEWPORT_W, VIEWPORT_H)
		)

		pygame.transform.scale(self.surf, (SCALE, SCALE))
		pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())

		for p in self.sky_polys:
			scaled_poly = []
			for coord in p:
				scaled_poly.append((coord[0] * SCALE, coord[1] * SCALE))
			pygame.draw.polygon(self.surf, (0, 0, 0), scaled_poly)
			gfxdraw.aapolygon(self.surf, scaled_poly, (0, 0, 0))

		self.all_particles = []

		for i in range(self.n_landers):
			if self.landers[i].lander is None:
				continue
			self.all_particles += self.landers[i].particles

		for obj in self.all_particles:
			obj.ttl -= 0.15
			obj.color1 = (
				int(max(0.2, 0.15 + obj.ttl) * 255),
				int(max(0.2, 0.5 * obj.ttl) * 255),
				int(max(0.2, 0.5 * obj.ttl) * 255),
			)
			obj.color2 = (
				int(max(0.2, 0.15 + obj.ttl) * 255),
				int(max(0.2, 0.5 * obj.ttl) * 255),
				int(max(0.2, 0.5 * obj.ttl) * 255),
			)

		for obj in self.all_particles + self.drawlist:
			for f in obj.fixtures:
				trans = f.body.transform
				if type(f.shape) is circleShape:
					pygame.draw.circle(
						self.surf,
						color=obj.color1,
						center=trans * f.shape.pos * SCALE,
						radius=f.shape.radius * SCALE,
					)
					pygame.draw.circle(
						self.surf,
						color=obj.color2,
						center=trans * f.shape.pos * SCALE,
						radius=f.shape.radius * SCALE,
					)

				else:
					path = [trans * v * SCALE for v in f.shape.vertices]
					pygame.draw.polygon(self.surf, color=obj.color1, points=path)
					gfxdraw.aapolygon(self.surf, path, obj.color1)
					pygame.draw.aalines(
						self.surf, color=obj.color2, points=path, closed=True
					)

				for x in [self.helipad_x1, self.helipad_x2]:
					x = x * SCALE
					flagy1 = self.helipad_y * SCALE
					flagy2 = flagy1 + 50
					pygame.draw.line(
						self.surf,
						color=(255, 255, 255),
						start_pos=(x, flagy1),
						end_pos=(x, flagy2),
						width=1,
					)
					pygame.draw.polygon(
						self.surf,
						color=(204, 204, 0),
						points=[
							(x, flagy2),
							(x, flagy2 - 10),
							(x + 25, flagy2 - 5),
						],
					)
					gfxdraw.aapolygon(
						self.surf,
						[(x, flagy2), (x, flagy2 - 10), (x + 25, flagy2 - 5)],
						(204, 204, 0),
					)

		self.surf = pygame.transform.flip(self.surf, False, True)

		if self.render_mode == "human":
			assert self.screen is not None
			self.screen.blit(self.surf, (0, 0))
			pygame.event.pump()
			self.clock.tick(self.metadata["render_fps"])
			pygame.display.flip()
		elif self.render_mode == "rgb_array":
			return np.transpose(
				np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
			)

	def _generate_moon(self):

		# terrain
		CHUNKS = 11
		height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
		chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
		self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
		self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
		self.helipad_y = H / 4
		height[CHUNKS // 2 - 2] = self.helipad_y
		height[CHUNKS // 2 - 1] = self.helipad_y
		height[CHUNKS // 2 + 0] = self.helipad_y
		height[CHUNKS // 2 + 1] = self.helipad_y
		height[CHUNKS // 2 + 2] = self.helipad_y
		smooth_y = [
			0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
			for i in range(CHUNKS)
		]

		self.moon = self.world.CreateStaticBody(
			shapes=edgeShape(vertices=[(0, 0), (W, 0)])
		)
		self.sky_polys = []
		for i in range(CHUNKS - 1):
			p1 = (chunk_x[i], smooth_y[i])
			p2 = (chunk_x[i + 1], smooth_y[i + 1])
			self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
			self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

		self.moon.color1 = (0.0, 0.0, 0.0)
		self.moon.color2 = (0.0, 0.0, 0.0)