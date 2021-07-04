import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def angleReward(obs, r):
    return np.float32(r + 10*np.abs(obs[2]))  # angle reward


class Trajectory(object):
    """
    Store 1 trajectory of transitions with log π(a|s)
    """
    def __init__(self, maxStep, gamma, dimState):
        self.maxStep = maxStep
        self.gamma = gamma
        self.dimState = dimState
        self.alloc()

    def __len__(self):
        return self.filled

    def alloc(self):
        self.state_buff = np.zeros((self.maxStep, self.dimState),
                                   dtype=np.float32)
        self.act_buff = np.zeros(self.maxStep, dtype=np.int64)
        self.rew_buff = np.zeros(self.maxStep, dtype=np.float32)
        self.next_state_buff = np.zeros((self.maxStep, self.dimState),
                                        dtype=np.float32)
        self.mask_buff = np.zeros(self.maxStep, dtype=np.uint8)
        self.log_prob_buff = np.zeros(self.maxStep, dtype=np.float32)
        self.ret_buff = np.zeros(self.maxStep, dtype=np.float32)
        self.clear()

    def clear(self):
        self.filled = 0
        self.position = 0


    # store 1 transition with log π(a|s)
    def push(self, s, a, r, sp, mask, log_p):
        self.state_buff[self.position] = s
        self.act_buff[self.position] = a
        self.rew_buff[self.position] = r
        self.next_state_buff[self.position] = sp
        self.mask_buff[self.position] = mask
        self.log_prob_buff[self.position] = log_p
        self.position = (self.position + 1) % self.maxStep

        if self.filled < self.maxStep:
            self.filled += 1

    # get data of whole trajectory with return-to-go
    def get_trajectory(self):
        s = torch.FloatTensor(self.state_buff[:self.filled])
        a = torch.LongTensor(self.act_buff[:self.filled])
        r = torch.FloatTensor(self.rew_buff[:self.filled])
        sp = torch.FloatTensor(self.next_state_buff[:self.filled])
        mask = torch.Tensor(self.mask_buff[:self.filled])
        ret = torch.FloatTensor(self.ret_buff[:self.filled])
        log_p = torch.FloatTensor(self.log_prob_buff[:self.filled])
        return s, a, r, sp, mask, ret, log_p

    # get mini-batch data with return-to-go and log π(a|s)
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(self.filled), size=batch_size,
                               replace=True)
        s = torch.FloatTensor(self.state_buff[idx])
        a = torch.LongTensor(self.act_buff[idx])
        r = torch.FloatTensor(self.rew_buff[idx])
        sp = torch.FloatTensor(self.next_state_buff[idx])
        mask = torch.Tensor(self.mask_buff[idx])
        ret = torch.FloatTensor(self.ret_buff[idx])
        log_p = torch.FloatTensor(self.log_prob_buff[idx])
        return s, a, r, sp, mask, ret, log_p


class MLPNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        out = self.fc3(h)
        return out


class AgentREINFORCE:
    """REINFORCE agent"""
    def __init__(self, env, maxStep=2000):
        self.name = "REINFORCE"

        # Environment information
        self.env = env
        self.dimState = env.observation_space.shape[0]
        self.dimAction = env.action_space.n
        self.gamma = 0.99
        self.maxStep = maxStep

        # Rollout buffer
        self.memory = Trajectory(self.maxStep, self.gamma, self.dimState)

        # Policy network & optimizer
        hidden_dim = 64
        self.pnet = MLPNet(self.dimState, self.dimAction, hidden_dim)
        self.policy_lr = 5e-3
        self.policy_optimizer = Adam(self.pnet.parameters(), lr=self.policy_lr)

    def save(self):
        torch.save(self.pnet.state_dict(), f"{self.name}_saved_policy.pt")
        if 'vnet' in dir(self):
            torch.save(self.vnet.state_dict(), f"{self.name}_saved_value.pt")

    def load(self):
        self.pnet.load_state_dict(torch.load(f"{self.name}_saved_policy.pt"))
        if 'vnet' in dir(self):
            self.vnet.load_state_dict(torch.load(f"{self.name}_saved_value.pt"))

    def getPolicy(self, state):  # get π(.|s), current policy given state
        logit = self.pnet(state)
        prob = F.softmax(logit, -1)
        policy = Categorical(prob)
        return policy

    def getAction(self, state):  # sample action following current policy
        policy = self.getPolicy(state)
        action = policy.sample()
        log_prob = policy.log_prob(action)
        return action.item(), log_prob.item()

    def log_prob(self, state, action):  # log π(a|s) based on current policy
        policy = self.getPolicy(state)
        return policy.log_prob(action)

    def final_value(self, finalIdx):  # calculate final value
        done = self.memory.mask_buff[finalIdx]
        if done:
            return 0
        if self.gamma == 1:
            factor = 100
        else:
            factor = 1 / (1-self.gamma)
        finalValue = self.memory.rew_buff[finalIdx] * factor
        return finalValue

    def calc_return(self):  # calculate return-to-go for each t
        finalIdx = self.memory.filled - 1
        g = self.final_value(finalIdx)
        for t in range(finalIdx, -1, -1):
            g = self.memory.rew_buff[t] + self.gamma*g
            self.memory.ret_buff[t] = g

    # Simulate 1 episode using current policy and return a list of rewards
    # Training is executed after the episode ends
    def runEpisode(self, maxStep, render=False, test=False):
        s = self.env.reset()
        done = False
        rewards = []
        for et_i in range(maxStep):
            if render:
                self.env.render()
            s_tensor = torch.FloatTensor(s)
            a, log_p = self.getAction(s_tensor)
            sp, r, done, _ = self.env.step(a)
            r = angleReward(sp, r)
            rewards.append(r)
            if not test:
                self.memory.push(s, a, r, sp, done, log_p)
            s = sp
            if done:
                break
        if not test:
            self.calc_return()
            self.train()
            self.memory.clear()
        return rewards

    # Test the current policy and return a sum of rewards
    def runTest(self, maxStep=1000, render=True):
        rewards = self.runEpisode(maxStep=maxStep, render=render,
                                  test=True)
        ret, nStep = sum(rewards), len(rewards)
        print(f"Test episode, return = {ret:.1f} in {nStep} steps")
        return ret

    # Run multiple episodes to train the agent and give a learning plot
    def runMany(self, nEpisode=1000):
        retHist = []
        max5 = 0
        for ep_i in range(nEpisode):
            rewards = self.runEpisode(maxStep=self.maxStep)
            ret = sum(rewards)
            print(f"Train episode i={ep_i}, return = {ret:.1f}")
            retHist.append(ret)
            if ep_i < 4:
                continue
            avg5 = sum(retHist[-5:])/5  # average return of the last 5 episodes
            if avg5 > max5 and max(retHist[-5:]) == retHist[-1]:
                max5 = avg5
                print(f"iter {ep_i}, avg5 = {avg5:.1f} updated")
                if avg5 > 100:
                    self.save()
                if avg5 >= self.maxStep:
                    print(f"Average reached {self.maxStep} after {ep_i} episodes. Solution found")
                    break
        plt.plot(retHist)
        plt.show()

    def train(self):
        self.policy_optimizer.zero_grad()
        s, a, r, sp, mask, ret, _ = self.memory.get_trajectory()
        #######################################################################
        # Task 1: Train REINFORCE agent
        # 1) Compute log π(a|s) using "s", "a" and "self.log_prob"
        # 2) Compute policy gradient loss using log π(a|s) and "ret"
        #    and assign it to "policy_loss"
        #######################################################################
        print(a)
        log_p = self.log_prob(s, a)
        gamma_t = np.zeros(len(log_p))
        for t in range(len(log_p)):
            gamma_t[t] = self.gamma ** t
        gamma_t = torch.FloatTensor(gamma_t)
        # print(log_p, ret)
        policy_loss = - torch.sum(log_p * gamma_t * ret)
        policy_loss.backward()
        clip_grad_norm_(self.pnet.parameters(), 1.0)
        self.policy_optimizer.step()


class AgentREINFORCEwBaseline(AgentREINFORCE):
    """REINFORCE with baseline agent"""
    def __init__(self, env, maxStep=2000):
        super(AgentREINFORCEwBaseline, self).__init__(env, maxStep)
        self.name = "REINFORCE_baseline"

        # Value network & optimizer
        hidden_dim = 64
        self.vnet = MLPNet(self.dimState, 1, hidden_dim)
        self.value_lr = 1e-2
        self.value_optimizer = Adam(self.vnet.parameters(),
                                    lr=self.value_lr)

    def getValue(self, state):  # get V(s)
        return self.vnet(state).squeeze()

    def final_value(self, finalIdx):  # calculate final value using baseline
        done = self.memory.mask_buff[finalIdx]
        if done:
            return 0
        sp = self.memory.next_state_buff[finalIdx]
        sp_tensor = torch.FloatTensor(sp)
        finalValue = self.getValue(sp_tensor).item()
        return finalValue

    def train(self):
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        s, a, r, sp, mask, ret, _ = self.memory.get_trajectory()
        #######################################################################
        # Task 2: Train REINFORCE with baseline agent
        # 1) Compute V(s) using "s" and "self.getValue"
        # 2) Compute log π(a|s) using "s", "a" and "self.log_prob"
        # 3) Compute advantage using V(s) and "ret"
        #    (It must not have gradient. You may use ".detach()")
        # 4) Compute policy gradient loss using log π(a|s) and advantage
        #    and assign it to "policy_loss"
        # 5) Compute value loss using V(s) and "ret"
        #    and assign it to "value_loss"
        #######################################################################
        val = self.getValue(s)
        log_p = self.log_prob(s, a)
        adv = ret - val.detach()
        gamma_t = np.zeros(len(log_p))
        for t in range(len(log_p)):
            gamma_t[t] = self.gamma ** t
        gamma_t = torch.FloatTensor(gamma_t)
        policy_loss = - torch.sum(log_p * gamma_t * adv)
        value_loss = 1/len(val) * torch.sum(torch.square(ret - val))
        policy_loss.backward()
        clip_grad_norm_(self.pnet.parameters(), 1.0)
        self.policy_optimizer.step()
        value_loss.backward()
        clip_grad_norm_(self.vnet.parameters(), 1.0)
        self.value_optimizer.step()


class AgentTDAC(AgentREINFORCEwBaseline):
    """TD actor-critic agent"""
    def __init__(self, env, maxStep=2000):
        super(AgentTDAC, self).__init__(env, maxStep)
        self.name = "TDAC"

    def train(self):
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        s, a, r, sp, mask, _, _ = self.memory.get_trajectory()
        #######################################################################
        # Task 3: Train TD advantage actor-critic agent
        # 1) Compute V(s) and log π(a|s)
        # 2) Compute V(s') using "sp" and "self.getValue"
        # 3) Compute 1-step TD target and TD advantage using 
        #    "r", "self.gamma", "mask", V(s) and V(s')
        #    (They must not have gradient. You may use ".detach()")
        # 4) Compute policy gradient loss using log π(a|s) and advantage
        #    and assign it to "policy_loss"
        # 5) Compute MSE loss of value using V(s) and TD target
        #    and assign it to "value_loss"
        #######################################################################
        val = self.getValue(s)
        next_val = self.getValue(sp)
        log_p = self.log_prob(s, a)
        target = r + self.gamma * next_val * (1 - mask)
        policy_loss = - torch.sum(log_p * (target.detach() - val.detach()))
        value_loss = 1/len(val)*torch.sum(torch.square(target.detach() - val))
        policy_loss.backward()
        clip_grad_norm_(self.pnet.parameters(), 1.0)
        self.policy_optimizer.step()
        value_loss.backward()
        clip_grad_norm_(self.vnet.parameters(), 1.0)
        self.value_optimizer.step()


class AgentPPO(AgentREINFORCEwBaseline):
    """Proximal policy optimization agent"""
    def __init__(self, env, maxStep=2000):
        super(AgentPPO, self).__init__(env, maxStep)
        self.name = "PPO"

        self.batch_size = 64
        self.num_update = maxStep // self.batch_size
        self.clip_ratio = 0.2

    def train(self):
        for epoch in range(self.num_update):
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            s, a, r, sp, mask, ret, curr_log_p = self.memory.sample(self.batch_size)
            ###################################################################
            # Task 4: Train PPO agent
            # 1) Compute V(s) and log π(a|s)
            # 2) Compute advantage using V(s) and "ret"
            #    (It must not have gradient. You may use ".detach()")
            # 3) Compute policy ratio using log π(a|s) and "curr_log_p"
            # 4) Clip the ratio using self.clip_ratio
            # 5) Compute policy loss using ratio, clipped ratio, and advantage
            #    and assign it to "policy_loss"
            # 6) Compute value loss using V(s) and "ret"
            #    and assign it to "value_loss"
            ###################################################################
            val = self.getValue(s)
            log_p = self.log_prob(s, a)
            adv = ret - val.detach()
            ratio = torch.exp(log_p - curr_log_p)
            clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            # print(val.shape, s.shape, s)
            policy_loss = -torch.sum(torch.min(ratio * adv, clipped * adv))/self.batch_size
            value_loss = torch.sum(torch.square(ret - val))/self.batch_size
            policy_loss.backward()
            clip_grad_norm_(self.pnet.parameters(), 1.0)
            self.policy_optimizer.step()
            value_loss.backward()
            clip_grad_norm_(self.vnet.parameters(), 1.0)
            self.value_optimizer.step()


def run(AgentClass=AgentREINFORCE, mode='train'):
    env = gym.make('CartPole-v1').unwrapped
    agent = AgentClass(env)
    if mode == 'train':
        # agent.runEpisode(2000, test=False)
        agent.runMany(10)
    elif mode == 'test':
        agent.load()
    agent.runTest()
    env.close()


if __name__ == '__main__':
    """
    1) Select the agent you want to train or test
    2) Select wheter to train the agent
       or test the best agent you already trained
    """
    agentClass = AgentREINFORCE
    # agentClass = AgentREINFORCEwBaseline
    # agentClass = AgentTDAC
    # agentClass = AgentPPO

    mode = "train"
    # mode = "test"

    run(agentClass, mode)
