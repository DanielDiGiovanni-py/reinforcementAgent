
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
    self.a = a
    self.b_start = b_start
    self.b_frames = b_frames
    self.frame = 1  
    self.batch_size = batch_size
    self.buffer_size = buffer_size
    self.buffer = []
    self.pos = 0
    self.priorities = np.zeros((buffer_size,), dtype=np.float32)
    self.seed = np.random.seed(seed)
    self.n_step = n_step
    self.parallel_env = parallel_env
    self.n_step_buffer = [deque(maxlen=self.n_step) for i in range(parallel_env)]
    self.iter_ = 0
    self.g = g

  def calc_multistep_return(self, n_step_buffer):
    Return = 0
    for idx in range(self.n_step):
      # check this later
      Return += self.g ** idx * n_step_buffer[idx][2]

    return n_step_buffer[0][0], n_step_buffer[0][1], Return, n_step_buffer[-1][3], n_step_buffer[-1][4]

  def beta_by_frame(self, frame_idx):
    """
    Based on paper: PRIORITIZED EXPERIENCE REPLAY
    "The estimation of the expected value with stochastic updates relies on those updates corresponding
    to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
    distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
    converge to (even if the policy and state distribution are fixed). We can correct this bias by using
    importance-sampling (IS) weights"

    """
    return min(1.0, self.b_start + frame_idx * (1.0 - self.b_start) / self.b_frames)

  def add(self, state, action, reward, next_state, done):
    if self.iter_ == self.parallel_env:
      self.iter_ = 0
    assert state.ndim == next_state.ndim
    state = np.expand_dims(state, 0)
    next_state = np.expand_dims(next_state, 0)

    # n_step calc
    self.n_step_buffer[self.iter_].append((state, action, reward, next_state, done))
    if len(self.n_step_buffer[self.iter_]) == self.n_step:
      state, action, reward, next_state, done = self.calc_multistep_return(self.n_step_buffer[self.iter_])

    max_prio = self.priorities.max() if self.buffer else 1.0  # gives max priority if buffer is not empty else 1

    if len(self.buffer) < self.buffer_size:
      self.buffer.append((state, action, reward, next_state, done))
    else:
      # puts the new data on the position of the old ones since it circles via position variable
      # since if len(buffer) == buffer_size -> pos == 0 -> oldest memory
      self.buffer[self.pos] = (state, action, reward, next_state, done)

    self.priorities[self.pos] = max_prio
    self.pos = (self.pos + 1) % self.buffer_size  # lets the pos circle in the ranges of buffer_size if pos+1 > cap --> new posi = 0
    self.iter_ += 1

  def sample(self):
    N = len(self.buffer)
    if N == self.buffer_size:
      prio = self.priorities
    else:
      prio = self.priorities[:self.pos]

    # calc P = p^a/sum(p^a)
    probs = prio ** self.a
    P = probs / probs.sum()

    # gets the indices depending on the probability p
    indices = np.random.choice(N, self.batch_size, p=P)
    samples = [self.buffer[idx] for idx in indices]

    beta = self.beta_by_frame(self.frame)
    self.frame += 1

    # Compute importance-sampling weight
    weights = (N * P[indices]) ** (-beta)
    # normalize weights
    weights /= weights.max()
    weights = np.array(weights, dtype=np.float32)

    states, actions, rewards, next_states, dones = zip(*samples)
    return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones, indices, weights

  def update_priorities(self, batch_indices, batch_priorities):
    for idx, prio in zip(batch_indices, batch_priorities):
      self.priorities[idx] = prio

  def __len__(self):
    return len(self.buffer)

def weight_init(layers):
  for layer in layers:
    torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

class DQN(nn.Module):
  def __init__(self, state_space, action_space, layer_size):
    super(DQN, self).__init__()
    self.seed = torch.manual_seed(1)
    self.input_shape = state_space
    self.action_space = action_space

    self.input_layer = nn.Linear(self.input_shape, layer_size)
    self.layer_1 = nn.Linear(layer_size, layer_size)
    self.layer_2 = nn.Linear(layer_size, layer_size)
    self.layer_3 = nn.Linear(layer_size, layer_size)
    self.layer_4 = nn.Linear(layer_size, layer_size)
    self.layer_5 = nn.Linear(layer_size, action_space)
    weight_init([self.input_layer, self.layer_1, self.layer_2, self.layer_3, self.layer_4, self.layer_5])

  def forward(self, input):
    x = torch.relu(self.input_layer(input))
    x = torch.relu(self.layer_1(x))
    x = torch.relu(self.layer_2(x))
    x = torch.relu(self.layer_3(x))
    x = torch.relu(self.layer_4(x))

    out = self.layer_5(x)

    return out


class Agent():
  def __init__(self, env_specs):
    """Initialize an Agent object.

    """
    self.state_space = env_specs['scent_space'].shape[0] + env_specs['feature_space'].shape[0]
    self.action_space = env_specs['action_space'].n
    self.seed = random.seed(0)
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    self.eps = 0.05

    # Q-Network
    print()
    self.qnetwork_local = DQN(self.state_space, self.action_space, self.layer_size).to(self.device)
    self.qnetwork_target = DQN(self.state_space, self.action_space, self.layer_size).to(self.device)

    self.optimizer = optim.SGD(self.qnetwork_local.parameters(), lr=self.LR, momentum=0.9, weight_decay=0.000001)
    print(self.qnetwork_local)

    # Replay memory
    self.memory = PrioritizedReplay(self.BUFFER_SIZE, self.BATCH_SIZE, seed=self.seed, g=self.g, n_step=self.n_step,
                                    parallel_env=self.worker)

    # Initialize time step
    self.t_step = 0

  def update(self, state, action, reward, next_state, done, timestep):
    # Save experience in replay memory
    self.memory.add(np.concatenate((state[0], state[2])), action, reward, np.concatenate((next_state[0], next_state[2])), done)
    # Learn every UPDATE_EVERY time steps.
    self.t_step = timestep % self.UPDATE_EVERY
    if self.t_step == 0:
      # If there are enough samples, then get a random subset and learn
      if len(self.memory) > self.BATCH_SIZE:
        experiences = self.memory.sample()
        loss = self.learn(experiences)
        self.Q_updates += 1

  def save(self, path):
    torch.save(self.qnetwork_local.state_dict(), path)

  def load_weights(self, root_path):
    self.qnetwork_local.load_state_dict(torch.load(root_path+'weights.pth'))
    self.qnetwork_local.eval()

  def act(self, state, mode='eval'):
    """Returns actions for given state as per current policy.
    
    """
    if mode== 'train':
      # Epsilon-greedy action selection
      if random.random() > self.eps:  # select greedy action if random number is higher than epsilon
        self.eps *= .999998
        state = np.concatenate((state[0], state[2]))
        state = torch.from_numpy(state).float().to(self.device)
  
        self.qnetwork_local.eval()
        with torch.no_grad():
          action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        action = np.argmax(action_values.cpu().data.numpy())
        return action
  
      else:
          action = random.choices(np.arange(self.action_space), k=self.worker)
          return action[0]
        
    if mode == 'eval':
      state = np.concatenate((state[0], state[2]))
      state = torch.from_numpy(state).float().to(self.device)
  
      self.qnetwork_local.eval()
      with torch.no_grad():
        action_values = self.qnetwork_local(state)
      self.qnetwork_local.train()
      action = np.argmax(action_values.cpu().data.numpy())
      return action


  def soft_update(self, local_model, target_model):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


  def learn(self, experiences):
    """Update value parameters using given batch of experience tuples.
    
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
    # Compute loss
    td_error = Q_targets - Q_expected
    loss = (td_error.pow(2) * weights).mean().to(self.device)
    # Minimize the loss
    loss.backward()
    clip_grad_norm_(self.qnetwork_local.parameters(), 1)
    self.optimizer.step()

    # update target network
    self.soft_update(self.qnetwork_local, self.qnetwork_target)
    # update per priorities
    self.memory.update_priorities(idx, abs(td_error.data.cpu().numpy()))

    return loss.detach().cpu().numpy()
