import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import numpy as np
from collections import deque
import random
import torch
import torch.optim as optim
import math


class PrioritizedReplay(object):
  """
  Proportional Prioritization Replay Experience
  """

  def __init__(self, buffer_size, batch_size, seed, g=0.99, n_step=1, a=0.6, b_start=0.4, b_frames=100000,
               parallel_env=4):
    # a = alpha (how much prioritization is used)
    # b_start = initial beta (importance-sampling correction)
    # b_frames = how many frames over which beta is annealed
    self.a = a
    self.b_start = b_start
    self.b_frames = b_frames
    self.frame = 1  # counter for frames to compute beta
    self.batch_size = batch_size
    self.buffer_size = buffer_size
    self.buffer = []  # replay buffer to store experiences
    self.pos = 0  # current position in the buffer (for circular overwrite)
    self.priorities = np.zeros((buffer_size,), dtype=np.float32)  # array of priorities
    self.seed = np.random.seed(seed)
    self.n_step = n_step
    self.parallel_env = parallel_env
    # n_step_buffer is a list of deques (one per parallel environment),
    # each storing up to n_step transitions for multi-step returns.
    self.n_step_buffer = [deque(maxlen=self.n_step) for i in range(parallel_env)]
    self.iter_ = 0  # tracks which parallel_env index to add to
    self.g = g  # discount factor

  def calc_multistep_return(self, n_step_buffer):
    """
    Calculates the n-step return for one of the parallel_env buffers.
    Returns the (state, action, accumulated_reward, next_state, done) tuple.
    """
    Return = 0
    for idx in range(self.n_step):
      # accumulate discounted rewards
      Return += self.g ** idx * n_step_buffer[idx][2]

    # n_step_buffer structure: (state, action, reward, next_state, done)
    return n_step_buffer[0][0], n_step_buffer[0][1], Return, n_step_buffer[-1][3], n_step_buffer[-1][4]

  def beta_by_frame(self, frame_idx):
    """
    Computes the current beta (importance-sampling correction parameter) by frame.
    Beta increases linearly from b_start to 1 over b_frames frames.
    """
    return min(1.0, self.b_start + frame_idx * (1.0 - self.b_start) / self.b_frames)

  def add(self, state, action, reward, next_state, done):
    """
    Adds a new experience to the replay buffer.
    Uses an n-step buffer to accumulate multi-step returns before storing.
    """
    if self.iter_ == self.parallel_env:
      # if we reached the number of parallel environments, reset
      self.iter_ = 0
    assert state.ndim == next_state.ndim
    # expand dims for consistent shape
    state = np.expand_dims(state, 0)
    next_state = np.expand_dims(next_state, 0)

    # store transition in the n_step_buffer for the current environment
    self.n_step_buffer[self.iter_].append((state, action, reward, next_state, done))
    # if the buffer for this env has n_step transitions, compute multi-step return
    if len(self.n_step_buffer[self.iter_]) == self.n_step:
      state, action, reward, next_state, done = self.calc_multistep_return(self.n_step_buffer[self.iter_])

    max_prio = self.priorities.max() if self.buffer else 1.0  # if buffer empty => priority=1.0

    if len(self.buffer) < self.buffer_size:
      # if buffer not yet full, append new experience
      self.buffer.append((state, action, reward, next_state, done))
    else:
      # otherwise overwrite oldest experience in a circular manner
      self.buffer[self.pos] = (state, action, reward, next_state, done)

    self.priorities[self.pos] = max_prio
    # move the position index forward (wrap-around if needed)
    self.pos = (self.pos + 1) % self.buffer_size
    self.iter_ += 1

  def sample(self):
    """
    Samples a batch of experiences from the buffer based on priority.
    Returns states, actions, rewards, next_states, dones, indices, weights.
    """
    N = len(self.buffer)
    if N == self.buffer_size:
      prio = self.priorities
    else:
      # if buffer not full, consider only valid priorities
      prio = self.priorities[:self.pos]

    # compute normalized probabilities P = p^a / sum(p^a)
    probs = prio ** self.a
    P = probs / probs.sum()

    # sample indices according to probability distribution P
    indices = np.random.choice(N, self.batch_size, p=P)
    samples = [self.buffer[idx] for idx in indices]

    # update beta based on current frame
    beta = self.beta_by_frame(self.frame)
    self.frame += 1

    # compute importance-sampling weights
    weights = (N * P[indices]) ** (-beta)
    # normalize weights by max weight
    weights /= weights.max()
    weights = np.array(weights, dtype=np.float32)

    # unzip the experiences
    states, actions, rewards, next_states, dones = zip(*samples)
    return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones, indices, weights

  def update_priorities(self, batch_indices, batch_priorities):
    """
    Updates the priorities of the sampled transitions given new priorities
    (often based on TD error).
    """
    for idx, prio in zip(batch_indices, batch_priorities):
      self.priorities[idx] = prio

  def __len__(self):
    """
    Returns the current size of the buffer.
    """
    return len(self.buffer)


def weight_init(layers):
  """
  Initializes the weights of the given layers with Kaiming normal initialization.
  """
  for layer in layers:
    torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


class DQN(nn.Module):
  """
  A basic fully-connected DQN network with configurable layer size.
  """
  def __init__(self, state_space, action_space, layer_size):
    super(DQN, self).__init__()
    self.seed = torch.manual_seed(1)
    self.input_shape = state_space
    self.action_space = action_space

    # Define layers
    self.input_layer = nn.Linear(self.input_shape, layer_size)
    self.layer_1 = nn.Linear(layer_size, layer_size)
    self.layer_2 = nn.Linear(layer_size, layer_size)
    self.layer_3 = nn.Linear(layer_size, layer_size)
    self.layer_4 = nn.Linear(layer_size, layer_size)
    self.layer_5 = nn.Linear(layer_size, action_space)

    # Initialize weights
    weight_init([self.input_layer, self.layer_1, self.layer_2, self.layer_3, self.layer_4, self.layer_5])

  def forward(self, input):
    """
    Forward pass: applies ReLU activations through hidden layers, then outputs Q-values.
    """
    x = torch.relu(self.input_layer(input))
    x = torch.relu(self.layer_1(x))
    x = torch.relu(self.layer_2(x))
    x = torch.relu(self.layer_3(x))
    x = torch.relu(self.layer_4(x))
    out = self.layer_5(x)
    return out


class Agent():
  """
  Agent class that holds the DQN networks (local/target) and performs training steps.
  """
  def __init__(self, env_specs):
    """Initialize an Agent object.
       env_specs is a dict containing environment info like action space, state space, etc.
    """
    # combine some environment state components
    self.state_space = env_specs['scent_space'].shape[0] + env_specs['feature_space'].shape[0]
    self.action_space = env_specs['action_space'].n
    self.seed = random.seed(0)
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # hyperparameters
    self.tau = 1e-3
    self.g = 0.99
    self.UPDATE_EVERY = 1
    self.worker = 1
    self.BUFFER_SIZE = int(1e4)
    self.BATCH_SIZE = 8 * self.worker
    self.LR = 0.00025
    self.Q_updates = 0
    self.layer_size = 8
    self.n_step = 10
    self.eps = 0.05  # epsilon for epsilon-greedy

    print()
    # create local and target networks
    self.qnetwork_local = DQN(self.state_space, self.action_space, self.layer_size).to(self.device)
    self.qnetwork_target = DQN(self.state_space, self.action_space, self.layer_size).to(self.device)

    # set optimizer
    self.optimizer = optim.SGD(self.qnetwork_local.parameters(), lr=self.LR, momentum=0.9, weight_decay=0.000001)
    print(self.qnetwork_local)

    # prioritized replay buffer
    self.memory = PrioritizedReplay(self.BUFFER_SIZE, self.BATCH_SIZE, seed=self.seed, g=self.g, n_step=self.n_step,
                                    parallel_env=self.worker)
    # time step tracker
    self.t_step = 0

  def update(self, state, action, reward, next_state, done, timestep):
    """
    Stores the new transition in replay memory. Triggers learning
    if conditions are met (enough samples & time step alignment).
    """
    # concatenate parts of the state
    self.memory.add(np.concatenate((state[0], state[2])), action, reward,
                    np.concatenate((next_state[0], next_state[2])), done)

    self.t_step = timestep % self.UPDATE_EVERY
    if self.t_step == 0:
      # only learn if enough samples in the buffer
      if len(self.memory) > self.BATCH_SIZE:
        experiences = self.memory.sample()
        loss = self.learn(experiences)
        self.Q_updates += 1

  def save(self, path):
    """Saves the local network's weights."""
    torch.save(self.qnetwork_local.state_dict(), path)

  def load_weights(self, root_path):
    """Loads pretrained weights into the local network."""
    self.qnetwork_local.load_state_dict(torch.load(root_path+'weights.pth'))
    self.qnetwork_local.eval()

  def act(self, state, mode='eval'):
    """
    Returns an action (int) based on the current policy (epsilon-greedy).
    If random < epsilon => random action, else => greedy.
    """
    if random.random() > self.eps:
      # reduce epsilon slightly
      self.eps *= .99998
      # combine relevant parts of state
      state = np.concatenate((state[0], state[2]))
      state = torch.from_numpy(state).float().to(self.device)

      self.qnetwork_local.eval()
      with torch.no_grad():
        action_values = self.qnetwork_local(state)
      self.qnetwork_local.train()

      action = np.argmax(action_values.cpu().data.numpy())
      return action
    else:
      if mode == 'eval':
        # pick random action from action space
        action = random.choices(np.arange(self.action_space), k=1)
      else:
        # if training, possibly pick multiple random actions?
        action = random.choices(np.arange(self.action_space), k=self.worker)
      return action[0]

  def soft_update(self, local_model, target_model):
    """
    Soft update model parameters:
    θ_target = τ*θ_local + (1 - τ)*θ_target
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

  def learn(self, experiences):
    """
    Processes a batch of experiences and updates the local Q-network.
    Also updates priorities in the replay buffer.
    """
    self.optimizer.zero_grad()
    states, actions, rewards, next_states, dones, idx, weights = experiences

    states = torch.FloatTensor(states).to(self.device)
    next_states = torch.FloatTensor(np.float32(next_states)).to(self.device)
    actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
    dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
    weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

    # Get max predicted Q values (for next states) from target model
    Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
    # Compute Q targets for current states
    Q_targets = rewards + (self.g ** self.n_step * Q_targets_next * (1 - dones))
    # Get expected Q values from local model
    Q_expected = self.qnetwork_local(states).gather(1, actions)
    # TD error
    td_error = Q_targets - Q_expected
    # weighted MSE loss with importance-sampling weights
    loss = (td_error.pow(2) * weights).mean().to(self.device)
    loss.backward()
    clip_grad_norm_(self.qnetwork_local.parameters(), 1)
    self.optimizer.step()

    # soft update target network
    self.soft_update(self.qnetwork_local, self.qnetwork_target)
    # update priorities in replay buffer with absolute TD errors
    self.memory.update_priorities(idx, abs(td_error.data.cpu().numpy()))

    return loss.detach().cpu().numpy()
