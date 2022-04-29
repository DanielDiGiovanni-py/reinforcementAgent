
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
  def __init__(self, state_space, action_space, layer_size, N_ATOMS=51, VMAX=10, VMIN=-10):
    super(DQN, self).__init__()
    self.seed = torch.manual_seed(1)
    self.input_shape = state_space
    self.action_space = action_space
    self.N_ATOMS = N_ATOMS
    self.VMAX = VMAX
    self.VMIN = VMIN
    self.DZ = (VMAX - VMIN) / (N_ATOMS - 1)

    self.input_layer = nn.Linear(self.input_shape, layer_size)
    self.layer_1 = nn.Linear(layer_size, layer_size)
    self.layer_2 = nn.Linear(layer_size, layer_size)
    self.layer_3 = nn.Linear(layer_size, layer_size)
    self.layer_4 = nn.Linear(layer_size, layer_size)
    self.layer_5 = nn.Linear(layer_size, action_space*N_ATOMS)
    weight_init([self.input_layer, self.layer_1, self.layer_2, self.layer_3, self.layer_4, self.layer_5])

    self.register_buffer("supports", torch.arange(VMIN, VMAX + self.DZ, self.DZ)) # basic value vector - shape n_atoms stepsize dz
    self.softmax = nn.Softmax(dim=1)

  def forward(self, input):
    m = nn.GELU()
    x = m(self.input_layer(input))
    x = m(self.layer_1(x))
    x = m(self.layer_2(x))
    x = m(self.layer_3(x))
    x = m(self.layer_4(x))

    q_distr = self.layer_5(x)
    prob = self.softmax(q_distr.view(-1, self.N_ATOMS)).view(-1, self.action_space, self.N_ATOMS)
    return prob

  def act(self, state):
    prob = self.forward(state).data.cpu()
    # create value distribution for each action - shape: (batch_size, action_space, 51)
    expected_value = prob.cpu() * self.supports.cpu()
    # sum up the prob*values for the action dimension - shape: (batch_size, action_space)
    actions = expected_value.sum(2)
    return actions

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
    self.eps = 0.5

    self.N_ATOMS = 51
    self.VMAX = 10
    self.VMIN = -10

    # Q-Network
    print()
    self.qnetwork_local = DQN(self.state_space, self.action_space, self.layer_size).to(self.device)
    self.qnetwork_target = DQN(self.state_space, self.action_space, self.layer_size).to(self.device)

    self.optimizer = optim.SGD(self.qnetwork_local.parameters(), lr=self.LR, momentum=0.9, weight_decay=0.00001)
    print(self.qnetwork_local)

    # Replay memory
    self.memory = PrioritizedReplay(self.BUFFER_SIZE, self.BATCH_SIZE, seed=self.seed, g=self.g, n_step=self.n_step,
                                    parallel_env=self.worker)

    # Initialize time step
    self.t_step = 0

  def projection_distribution(self, next_distr, next_state, rewards, dones):
    """
    """
    batch_size = next_state.size(0)
    # create support atoms
    delta_z = float(self.VMAX - self.VMIN) / (self.N_ATOMS - 1)
    support = torch.linspace(self.VMIN, self.VMAX, self.N_ATOMS)
    support = support.unsqueeze(0).expand_as(next_distr).to(self.device)

    rewards = rewards.expand_as(next_distr)
    dones = dones.expand_as(next_distr)

    ## Compute the projection of T̂ z onto the support {z_i}
    Tz = rewards + (1 - dones) * self.g ** self.n_step * support
    Tz = Tz.clamp(min=self.VMIN, max=self.VMAX)
    b = ((Tz - self.VMIN) / delta_z).cpu()  # .to(self.device)
    l = b.floor().long().cpu()  # .to(self.device)
    u = b.ceil().long().cpu()  # .to(self.device)

    offset = torch.linspace(0, (batch_size - 1) * self.N_ATOMS, batch_size).long() \
      .unsqueeze(1).expand(batch_size, self.N_ATOMS)
    # Distribute probability of T̂ z
    proj_dist = torch.zeros(next_distr.size())
    proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_distr.cpu() * (u.float() - b)).view(-1))
    proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_distr.cpu() * (b - l.float())).view(-1))

    return proj_dist

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
    if mode == 'train' or mode == 'eval':
      # Epsilon-greedy action selection
      if random.random() > self.eps:  # select greedy action if random number is higher than epsilon
        self.eps *= .999998
        state = np.concatenate((state[0], state[2]))
        state = torch.from_numpy(state).float().to(self.device)

        self.qnetwork_local.eval()
        with torch.no_grad():
          action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        action = np.argmax(np.argmax(action_values.cpu().data.numpy(), axis=2))
        return action

      else:
          action = random.choices(np.arange(self.action_space), k=self.worker)
          return action[0]

    #if mode == 'peach':
      #state = np.concatenate((state[0], state[2]))
      #state = torch.from_numpy(state).float().to(self.device)

      #self.qnetwork_local.eval()
      #with torch.no_grad():
        #action_values = self.qnetwork_local(state)
      #self.qnetwork_local.train()
      #action = np.argmax(np.argmax(action_values.cpu().data.numpy(), axis=2))
      #return action


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

    batch_size = self.BATCH_SIZE
    self.optimizer.zero_grad()
    # next_state distribution
    next_distr = self.qnetwork_target(next_states)
    next_actions = self.qnetwork_target.act(next_states)
    # chose max action indx
    next_actions = next_actions.max(1)[1].data.cpu().numpy()
    # gather best distr
    next_best_distr = next_distr[range(batch_size), next_actions]

    proj_distr = self.projection_distribution(next_best_distr, next_states, rewards, dones).to(self.device)

    # Compute loss
    # calculates the prob_distribution for the actions based on the given state
    prob_distr = self.qnetwork_local(states)
    actions = actions.unsqueeze(1).expand(batch_size, 1, self.N_ATOMS)
    # gathers the prob_distribution for the chosen action
    state_action_prob = prob_distr.gather(1, actions).squeeze(1)
    loss_prio = -((state_action_prob.log() * proj_distr.detach()).sum(dim=1).unsqueeze(1) * weights)  # at some point none values arise
    # print("LOSS: ",loss_prio)
    loss = loss_prio.mean()

    # Minimize the loss
    loss.backward()
    clip_grad_norm_(self.qnetwork_local.parameters(), 1)
    self.optimizer.step()

    # ------------------- update target network ------------------- #
    self.soft_update(self.qnetwork_local, self.qnetwork_target)
    # update per priorities
    self.memory.update_priorities(idx, abs(loss_prio.data.cpu().numpy()))
    return loss.detach().cpu().numpy()
