
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import numpy as np
from collections import deque, namedtuple
import random
import torch
import torch.optim as optim


class PrioritizedReplay(object):
  """
  Proportional Prioritization
  """

  def __init__(self, capacity, batch_size, seed, gamma=0.99, n_step=1, alpha=0.6, beta_start=0.4, beta_frames=100000,
               parallel_env=4):
    self.alpha = alpha
    self.beta_start = beta_start
    self.beta_frames = beta_frames
    self.frame = 1  # for beta calculation
    self.batch_size = batch_size
    self.capacity = capacity
    self.buffer = []
    self.pos = 0
    self.priorities = np.zeros((capacity,), dtype=np.float32)
    self.seed = np.random.seed(seed)
    self.n_step = n_step
    self.parallel_env = parallel_env
    self.n_step_buffer = [deque(maxlen=self.n_step) for i in range(parallel_env)]
    self.iter_ = 0
    self.gamma = gamma

  def calc_multistep_return(self, n_step_buffer):
    Return = 0
    for idx in range(self.n_step):
      Return += self.gamma ** idx * n_step_buffer[idx][2]

    return n_step_buffer[0][0], n_step_buffer[0][1], Return, n_step_buffer[-1][3], n_step_buffer[-1][4]

  def beta_by_frame(self, frame_idx):
    """
    Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.

    3.4 ANNEALING THE BIAS (Paper: PER)
    We therefore exploit the flexibility of annealing the amount of importance-sampling
    correction over time, by defining a schedule on the exponent
    that reaches 1 only at the end of learning. In practice, we linearly anneal from its initial value 0 to 1
    """
    return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

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

    if len(self.buffer) < self.capacity:
      self.buffer.append((state, action, reward, next_state, done))
    else:
      # puts the new data on the position of the oldes since it circles via pos variable
      # since if len(buffer) == capacity -> pos == 0 -> oldest memory (at least for the first round?)
      self.buffer[self.pos] = (state, action, reward, next_state, done)

    self.priorities[self.pos] = max_prio
    self.pos = (
                         self.pos + 1) % self.capacity  # lets the pos circle in the ranges of capacity if pos+1 > cap --> new posi = 0
    self.iter_ += 1

  def sample(self):
    N = len(self.buffer)
    if N == self.capacity:
      prios = self.priorities
    else:
      prios = self.priorities[:self.pos]

    # calc P = p^a/sum(p^a)
    probs = prios ** self.alpha
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

class DDQN(nn.Module):
  def __init__(self, state_size, action_size, layer_size, n_step, seed, layer_type="ff"):
    super(DDQN, self).__init__()
    self.seed = torch.manual_seed(1)
    self.input_shape = state_size
    self.action_size = action_size
    self.state_dim = len(state_size)
    if self.state_dim == 3:
      self.cnn_1 = nn.Conv2d(4, out_channels=32, kernel_size=8, stride=4)
      self.cnn_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
      self.cnn_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
      weight_init([self.cnn_1, self.cnn_2, self.cnn_3])

      self.ff_1 = nn.Linear(self.calc_input_layer(), layer_size)
      self.ff_2 = nn.Linear(layer_size, action_size)
      weight_init([self.ff_1])

    elif self.state_dim == 1:
      self.head_1 = nn.Linear(self.input_shape[0], layer_size)
      self.ff_1 = nn.Linear(layer_size, layer_size)
      self.ff_2 = nn.Linear(layer_size, action_size)
      weight_init([self.head_1, self.ff_1])
    else:
      print("Unknown input dimension!")

  def calc_input_layer(self):
    x = torch.zeros(self.input_shape).unsqueeze(0)
    x = self.cnn_1(x)
    x = self.cnn_2(x)
    x = self.cnn_3(x)
    return x.flatten().shape[0]

  def forward(self, input):
    """

    """
    if self.state_dim == 3:
      x = torch.relu(self.cnn_1(input))
      x = torch.relu(self.cnn_2(x))
      x = torch.relu(self.cnn_3(x))
      x = x.view(input.size(0), -1)
    else:
      x = torch.relu(self.head_1(input))

    x = torch.relu(self.ff_1(x))
    out = self.ff_2(x)

    return out


class Agent():
  '''The agent class that is to be filled.
     You are allowed to add any method you
     want to this class.
  '''

  # def load_weights(self, root_path):
  #   # Add root_path in front of the path of the saved network parameters
  #   # For example if you have weights.pth in the GROUP_MJ1, do `root_path+"weights.pth"` while loading the parameters
  #   pass

  """Interacts with and learns from the environment."""


  def __init__(self, env_specs):
    """Initialize an Agent object.

    Params
    ======
        state_size (int): dimension of each state
        action_size (int): dimension of each action
        Network (str): dqn network type
        layer_size (int): size of the hidden layer
        BATCH_SIZE (int): size of the training batch
        BUFFER_SIZE (int): size of the replay memory
        LR (float): learning rate
        TAU (float): tau for soft updating the network weights
        GAMMA (float): discount factor
        UPDATE_EVERY (int): update frequency
        device (str): device that is used for the compute
        seed (int): random seed
    """
    self.state_size = env_specs['feature_space'].shape
    self.action_size = env_specs['action_space'].n
    self.eta = .1
    self.seed = random.seed(1)
    self.t_seed = torch.manual_seed(1)
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.TAU = 1e-3
    self.GAMMA = 0.99
    self.UPDATE_EVERY = 1
    self.worker = 1
    self.BUFFER_SIZE = int(1e5)
    self.BATCH_SIZE = 1 * self.worker
    self.LR = 0.00025
    self.Q_updates = 0
    self.layer_size = 512
    self.n_step = 1

    self.action_step = 4
    self.last_action = None


    # Q-Network
    print()
    self.qnetwork_local = DDQN(self.state_size, self.action_size, self.layer_size, self.n_step, self.seed).to(self.device)
    self.qnetwork_target = DDQN(self.state_size, self.action_size, self.layer_size, self.n_step, self.seed).to(self.device)

    self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)
    print(self.qnetwork_local)

    # Replay memory
    self.memory = PrioritizedReplay(self.BUFFER_SIZE, self.BATCH_SIZE, seed=self.seed, gamma=self.GAMMA, n_step=self.n_step,
                                    parallel_env=self.worker)

    # Initialize time step (for updating every UPDATE_EVERY steps)
    self.t_step = 0


  def update(self, state, action, reward, next_state, done, timestep):
    # Save experience in replay memory
    self.memory.add(state[2], action, reward, next_state[2], done)

    # Learn every UPDATE_EVERY time steps.
    self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
    if self.t_step == 0:
      # If enough samples are available in memory, get random subset and learn
      if len(self.memory) > self.BATCH_SIZE:
        experiences = self.memory.sample()
        loss = self.learn_per(experiences)
        self.Q_updates += 1


  def act(self, state, eps=0., mode='eval'):
    """Returns actions for given state as per current policy. Acting only every 4 frames!

    Params
    ======
        frame: to adjust epsilon
        state (array_like): current state

    """

    # Epsilon-greedy action selection
    if random.random() > eps:  # select greedy action if random number is higher than epsilon
      state = np.array(state[2])
      if len(self.state_size) > 1:
        state = torch.from_numpy(state).float().to(self.device)
      else:
        state = torch.from_numpy(state).float().to(self.device)
      self.qnetwork_local.eval()
      with torch.no_grad():
        action_values = self.qnetwork_local(state)
      self.qnetwork_local.train()
      action = np.argmax(action_values.cpu().data.numpy())
      return action

    else:
      if mode == 'eval':
        action = random.choices(np.arange(self.action_size), k=1)
      else:
        action = random.choices(np.arange(self.action_size), k=self.worker)
      return action


  def soft_update(self, local_model, target_model):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)


  def learn_per(self, experiences):
    """Update value parameters using given batch of experience tuples.
    Params
    ======
        experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        gamma (float): discount factor
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
    Q_targets = rewards + (self.GAMMA ** self.n_step * Q_targets_next * (1 - dones))
    # Get expected Q values from local model
    Q_expected = self.qnetwork_local(states).gather(1, actions)
    # Compute loss
    td_error = Q_targets - Q_expected
    loss = (td_error.pow(2) * weights).mean().to(self.device)
    # Minimize the loss
    loss.backward()
    clip_grad_norm_(self.qnetwork_local.parameters(), 1)
    self.optimizer.step()

    # ------------------- update target network ------------------- #
    self.soft_update(self.qnetwork_local, self.qnetwork_target)
    # update per priorities
    self.memory.update_priorities(idx, abs(td_error.data.cpu().numpy()))

    return loss.detach().cpu().numpy()
