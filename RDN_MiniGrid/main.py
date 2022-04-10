import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.optim as optim

import gym 
from gym_minigrid.minigrid import *
from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

import time

from net import ACModel, RND

def discount_cumsum(x, discount):
    """
    Compute  cumulative sums of vectors.

    Input: [x0, x1, ..., xn]
    Output: [x0 + discount * x1 + discount^2 * x2 + ... , x1 + discount * x2 + ... , ... , xn]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def prepare_state(x):
    "Normalizes the state tensor and converts it to a PyTorch tensor"
    ns = torch.from_numpy(x).float().permute(2,0,1).unsqueeze(dim=0)
    maxv = ns.flatten().max()
    ns = ns / maxv
    return ns

env = ImgObsWrapper(gym.make('MiniGrid-DoorKey-8x8-v0'))
state = prepare_state(env.reset())

args = {
    'gamma': 0.99,
    'lmbda': 0.95,
    'eps_clip': 0.1,
    'entropy_beta' : 0.01,
    'lr': 5e-4,
    'lr_rnd': 5e-5, 
    'n_act': 5
}

def train(net, rnd, buffer):
    new_buffer = list(zip(*buffer))
    state_batch = torch.stack(new_buffer[0]).squeeze()
    rew_batch = np.array(new_buffer[1])
    new_state_batch = torch.stack(new_buffer[2]).squeeze()
    pi_batch = torch.cat(new_buffer[3])
    action_batch = np.array(new_buffer[4]) 

    h_in = new_buffer[5][0].detach()
    c_in = new_buffer[6][0].detach()
    h_out = new_buffer[7][0].detach()
    c_out = new_buffer[8][0].detach() 
    
    assert np.sum(np.array(new_buffer[9])) == 1
    done_mask = torch.FloatTensor(1 - np.array(new_buffer[9]))
  
    with torch.no_grad():
        _, v_prime, _, _ = net(new_state_batch, h_out, c_out)
    td_target = (torch.FloatTensor(rew_batch) + 
                 args['gamma'] * v_prime.squeeze() * done_mask.squeeze())
    for i in range(2):
        pi, v, _, _ = net(state_batch, h_in, c_in)   
        delta = td_target - v.squeeze()
        delta = delta.detach().numpy()
        advantage_lst = []
        advantage = 0.0
        for item in delta[::-1]:
            advantage = args['gamma'] * args['lmbda'] * advantage + item
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage = torch.FloatTensor(advantage_lst).squeeze()
        pi_a = pi.gather(1, torch.tensor(action_batch, dtype=torch.int64)).squeeze()       
        ratio = torch.exp(torch.log(pi_a) - torch.log(pi_batch))  # a/b == exp(log(a)-log(b))
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1-args['eps_clip'], 1+args['eps_clip']) * advantage

        # Entropy loss
        entropy = (torch.log(pi) * pi).sum(dim=1).mean()
        entropy_loss = args['entropy_beta'] * entropy

        loss = (-torch.min(surr1, surr2) + 
                F.smooth_l1_loss(v.squeeze(), td_target.detach().squeeze()) + 
                entropy_loss)

        optimizer.zero_grad()
        loss.mean().backward(retain_graph=False)
        optimizer.step() 

    loss_rnd = rnd.loss(state_batch)
    optimizer_rnd.zero_grad()
    loss_rnd.mean().backward(retain_graph=False)
    optimizer_rnd.step() 

    return net

action_map = {
    0 : 0, 
    1 : 1, 
    2 : 2, 
    3 : 3,
    4 : 5
}

net = ACModel(state.shape[1:], args['n_act'], 128, nn.Tanh())
rnd = RND(state.shape[1:])

optimizer = optim.Adam(net.parameters(), lr=args['lr'])
optimizer_rnd = optim.Adam(rnd.parameters(), lr=args['lr_rnd'])

all_rews = []

start = time.time()
for epoch in range(4500):
    h_in = torch.zeros([1, 1, 128])
    c_in = torch.zeros([1, 1, 128])    
    state = prepare_state(env.reset())
    done = False
    buffer = [] 
    counter = 0
    tot_rew = 0
    all_extra = 0
    while not done:
        counter += 1
        with torch.no_grad():
            pi, v, h_out, c_out = net(state, h_in, c_in)
        prob = Categorical(pi.squeeze())
        action = prob.sample().item()
        new_state, rew, done, _ = env.step(action_map[action])
        if rew == 0:
            rew = -0.1
        new_state = prepare_state(new_state)
        tot_rew += rew
        
        if counter >= 400:
            done = True
        with torch.no_grad():    
            extra_rew = 5*rnd.extra_reward(state)
            all_extra += extra_rew
        # Save in the buffer
        action = np.array([action])
        buffer.append([state, (rew + extra_rew)/10, new_state, 
                       pi[:,action[0]].detach(),
                       action, 
                       h_in, c_in, h_out, c_out, done])
        
        h_in, c_in = h_out, c_out
        state = new_state
    net = train(net, rnd, buffer)
    all_rews.append(tot_rew)
    if (epoch % 10 == 0) and (epoch > 0):
        print('Epoch:', epoch,
              'Avg Rew:', np.round(np.mean(all_rews[-10:]), 3),
              'Extra:', np.round(all_extra, 3))
    if np.mean(all_rews[-10:]) > -1.0:
        break
print('Total time:', time.time() - start)

torch.save(net.state_dict(), 'model_PPO_lstm_minigrid_curiosity_8x8_2.pt')
torch.save(optimizer.state_dict(), 'optim_PPO_lstm_minigrid_curiosity_8x8_2.pt')
