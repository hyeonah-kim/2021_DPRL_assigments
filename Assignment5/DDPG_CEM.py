import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cartpole import ContinuousCartPoleEnv
from copy import deepcopy
from matplotlib import pyplot as plt
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam


class ReplayBuffer(object):
    def __init__(self, maxlen, dimState):
        self.maxlen = maxlen
        self.dimState = dimState
        self.alloc()

    def __len__(self):
        return self.filled

    def alloc(self):
        self.state_buff = np.zeros((self.maxlen, self.dimState), 
                                   dtype=np.float32)
        self.act_buff = np.zeros((self.maxlen, 1), dtype=np.float32)
        self.rew_buff = np.zeros(self.maxlen, dtype=np.float32)
        self.next_state_buff = np.zeros((self.maxlen, self.dimState),
                                        dtype=np.float32)
        self.mask_buff = np.zeros(self.maxlen, dtype=np.uint8)
        self.clear()

    def clear(self):
        self.filled = 0
        self.position = 0

    def push(self, s, a, r, sp, mask):
        self.state_buff[self.position] = s
        self.act_buff[self.position] = a
        self.rew_buff[self.position] = r
        self.next_state_buff[self.position] = sp
        self.mask_buff[self.position] = mask
        self.position = (self.position + 1) % self.maxlen

        if self.filled < self.maxlen:
            self.filled += 1

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(self.filled), size=batch_size,
                               replace=True)
        s = torch.FloatTensor(self.state_buff[idx])
        a = torch.FloatTensor(self.act_buff[idx])
        r = torch.FloatTensor(self.rew_buff[idx])
        sp = torch.FloatTensor(self.next_state_buff[idx])
        mask = torch.Tensor(self.mask_buff[idx])
        return s, a, r, sp, mask


class MLPCritic(nn.Module):
    def __init__(self, dimState, dimAction, hidden_dim):
        super(MLPCritic, self).__init__()
        self.fc1 = nn.Linear(dimState+dimAction, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        out = self.fc3(h)
        return out


class LinActor(nn.Module):
    def __init__(self, dimState, dimAction, hidden_dim, act_limit):
        super(LinActor, self).__init__()
        self.fc1 = nn.Linear(dimState, dimAction, bias=False)
        self.act_limit = act_limit

    def forward(self, state):
        out = self.fc1(state)
        out = torch.tanh(out)
        return self.act_limit*out

    def set_params(self, params):
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())

            param.data.copy_(
                torch.from_numpy(params[cpt:cpt+tmp]).view(param.size()))
            cpt += tmp

    def get_params(self):
        return deepcopy(np.hstack([v.data.numpy().flatten() for v in
                                   self.parameters()]))

    def get_size(self):
        return self.get_params().shape[0]


class MLPActor(LinActor):
    def __init__(self, dimState, dimAction, hidden_dim, act_limit):
        super(MLPActor, self).__init__(dimState, dimAction, hidden_dim, act_limit)
        self.fc1 = nn.Linear(dimState, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, dimAction)

    def forward(self, state):
        h = F.relu(self.fc1(state))
        h = F.relu(self.fc2(h))
        out = torch.tanh(self.fc3(h))
        return self.act_limit*out


class AgentDDPG(object):
    def __init__(self, env, maxStep=2000):
        self.name = 'DDPG'

        # Environment information
        self.env = env
        self.dimState = env.observation_space.shape[0]
        self.dimAction = env.action_space.shape[0]
        self.act_limit = env.action_space.high[0]
        self.gamma = 0.99  # discount factor
        self.maxStep = maxStep

        # Experience replay buffer
        self.memory = ReplayBuffer(100000, self.dimState)
        self.batch_size = 64

        # Action noise
        self.noise_scale = 0.2

        # Network initialization
        hidden_dim = 64
        ActorClass = MLPActor
        # ActorClass = LinActor
        self.pnet = ActorClass(self.dimState, self.dimAction, hidden_dim, self.act_limit)
        self.target_pnet = ActorClass(self.dimState, self.dimAction, hidden_dim, self.act_limit)
        self.qnet = MLPCritic(self.dimState, self.dimAction, hidden_dim)
        self.target_qnet = MLPCritic(self.dimState, self.dimAction, hidden_dim)
        self.policy_lr = 5e-4
        self.policy_optimizer = Adam(self.pnet.parameters(), lr=self.policy_lr)
        self.value_lr = 1e-3
        self.value_optimizer = Adam(self.qnet.parameters(), lr=self.value_lr)
        self.hard_update()

        self.tau = 1e-2  # target network update rate

    def save(self):
        torch.save(self.pnet.state_dict(), f"{self.name}_saved_policy.pt")
        torch.save(self.qnet.state_dict(), f"{self.name}_saved_value.pt")

    def load(self):
        self.pnet.load_state_dict(torch.load(f"{self.name}_saved_policy.pt"))
        self.qnet.load_state_dict(torch.load(f"{self.name}_saved_value.pt"))

    def hard_update(self):
        for target, source in zip(self.target_pnet.parameters(),
                                  self.pnet.parameters()):
            target.data.copy_(source.data)
        for target, source in zip(self.target_qnet.parameters(),
                                  self.qnet.parameters()):
            target.data.copy_(source.data)

    def soft_update(self):
        for target, source in zip(self.target_pnet.parameters(),
                                  self.pnet.parameters()):
            average = target.data*(1. - self.tau) + source.data*self.tau
            target.data.copy_(average)
        for target, source in zip(self.target_qnet.parameters(),
                                  self.qnet.parameters()):
            average = target.data*(1. - self.tau) + source.data*self.tau
            target.data.copy_(average)

    def getAction(self, state, test=False):
        with torch.no_grad():
            action = self.pnet(state).item()
        if test:
            return action
        action += self.noise_scale * np.random.randn()
        return np.clip(action, -self.act_limit, self.act_limit)

    # Simulate 1 episode using current policy and return a list of rewards
    def runEpisode(self, maxStep, render=False, test=False):
        s = self.env.reset()
        done = False
        rewards = []
        for et_i in range(maxStep):
            if render:
                self.env.render()
            s_tensor = torch.FloatTensor(s)
            a = self.getAction(s_tensor, test)
            sp, r, done, _ = self.env.step(a)
            rewards.append(r)
            if not test:
                self.memory.push(s, a, r, sp, done)
                self.train()
                self.soft_update()
            s = sp
            if done:
                break
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
            print(f"Train episode i={ep_i+1}, return = {ret:.1f}")
            retHist.append(ret)
            if ep_i < 4:
                continue
            avg5 = sum(retHist[-5:])/5  # average return of the last 5 episodes
            if avg5 > max5 and max(retHist[-5:]) == retHist[-1]:
                max5 = avg5
                print(f"iter {ep_i+1}, avg5 = {avg5:.1f} updated")
                if avg5 > 100:
                    self.save()
                if avg5 >= self.maxStep:
                    print(f"Average reached {self.maxStep} after {ep_i+1} episodes. Solution found")
                    break
        plt.plot(retHist)
        plt.show()

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        s, a, r, sp, mask = self.memory.sample(self.batch_size)
        #######################################################################
        # Task 1-1: Train critic network
        # 1) Compute μ(s') using "sp" and "self.target_pnet"
        # 2) Compute TD target using "r", "self.gamma", "sp", "mask",
        #    "self.target_qnet", and μ(s')
        #    (It must not have gradient. You may use ".detach()")
        # 3) Compute Q(s, a) using "s", "a", and "self.qnet"
        # 4) Compute MSE loss of value using Q(s, a) and TD target
        #    and assign it to "value_loss"
        #######################################################################
        ap = self.target_pnet(sp)
        target = r + (1 - mask) * self.gamma * self.target_qnet(sp, ap).squeeze()
        value_loss = torch.sum(torch.square(target.detach() - self.qnet(s, a).squeeze())) / self.batch_size
        value_loss.backward()
        clip_grad_norm_(self.qnet.parameters(), 0.1)
        self.value_optimizer.step()
        
        #######################################################################
        # Task 1-2: Train actor network
        # 1) Compute μ(s) using "s" and "self.pnet"
        # 2) Compute policy gradient loss using "self.qnet", "s" and μ(s)
        #    assign it to "policy_loss"
        #######################################################################
        mu_s = self.pnet(s)
        policy_loss = - torch.sum(self.qnet(s, mu_s)) / self.batch_size
        policy_loss.backward()
        clip_grad_norm_(self.pnet.parameters(), 0.1)
        self.policy_optimizer.step()


class AgentCEM(object):
    def __init__(self, env, maxStep=1000, nPop=100, elitePortion=0.2):
        self.name = 'CEM'

        # Environment information
        self.env = env
        self.dimState = self.env.observation_space.shape[0]
        self.dimAction = env.action_space.shape[0]
        self.act_limit = env.action_space.high[0]
        self.maxStep = maxStep

        hidden_dim = 64
        # ActorClass = MLPActor
        ActorClass = LinActor
        self.pnet = ActorClass(self.dimState, self.dimAction, hidden_dim, self.act_limit)
        self.paramSize = self.pnet.get_size()
        self.mu = self.pnet.get_params()
        self.cov = np.ones(self.paramSize)
        self.nPop = nPop
        self.elitePortion = elitePortion
        self.nElite = int(self.nPop*self.elitePortion)
        self.paramSet = np.empty((self.nPop, self.paramSize), dtype=self.mu.dtype)
        self.scoreSet = np.zeros(self.nPop)

    def save(self):
        curr_params = self.pnet.get_params()
        best_params = self.get_best_param()
        self.pnet.set_params(best_params)
        torch.save(self.pnet.state_dict(), f"{self.name}_saved_policy.pt")
        self.pnet.set_params(curr_params)

    def load(self):
        self.pnet.load_state_dict(torch.load(f"{self.name}_saved_policy.pt"))

    # Simulate 1 episode using current policy and return a list of rewards
    def runEpisode(self, maxStep, render=False):
        s = self.env.reset()
        done = False
        rewards = []
        for et_i in range(maxStep):
            if render:
                self.env.render()
            s_tensor = torch.FloatTensor(s)
            with torch.no_grad():
                a = self.pnet(s_tensor).item()
            sp, r, done, _ = self.env.step(a)
            rewards.append(r)
            s = sp
            if done:
                break
        if not done:
            rewards.append(500)
        return rewards

    def runTest(self, maxStep=1000, render=True):
        rewards = self.runEpisode(maxStep=maxStep, render=render)
        ret, nStep = sum(rewards), len(rewards)
        print(f"Test episode, return = {ret:.1f} in {nStep} steps")
        return ret

    def runAgents(self, maxStep=1000):
        for a in range(self.nPop):
            params = self.paramSet[a, :]
            self.pnet.set_params(params)
            rewards = self.runEpisode(maxStep=maxStep)
            ret = sum(rewards)
            self.scoreSet[a] = ret

    def runMany(self, nGen=100):
        retHist = []
        max_eliteMin = 0
        for gen_i in range(nGen):
            self.sampleAgents()
            self.runAgents(maxStep=self.maxStep)
            rMean, rMax, rMin = self.updatePopParam()
            print (f"Train iteration {gen_i+1}, {self.elitePortion*100}% max, mean, min = ({rMax:.2f}, {rMean:.2f}, {rMin:.2f})")
            if self.paramSize < 10:
                print (f"mu  = {self.mu}")
                print (f"cov = {self.cov}")
            retHist.append(rMean)
            if rMin > max_eliteMin:
                max_eliteMin = rMin
                print(f"iter {gen_i+1}, elite minimum = {rMin:.1f} updated")
                if rMin > 100:
                    self.save()
                if rMin >= self.maxStep * 2.0:
                    print(f"Elite minimum reached {self.maxStep*2} after {gen_i+1} generations. Solution found")
                    break
        plt.plot(retHist)
        plt.show()

    def get_best_param(self):
        assert "best_idx" in dir(self)
        return self.paramSet[self.best_idx, :]

    def sampleAgents(self):
        #######################################################################
        # Task 2: Sample policy parameters from distribution
        # 1) Sample parameters from Gaussian dist. (with diagonal cov. matrix)
        #    using "self.mu" and "self.cov"
        # 2) Assign samples to "self.paramSet"
        #    (self.paramSet.shape = (self.nPop, self.paramSize))
        #######################################################################
        for i in range(self.nPop):
            e = np.random.normal(0.0, 1.0, self.paramSize)
            theta = self.mu + np.sqrt(self.cov) * e
            noise = np.random.normal(np.zeros(self.paramSize), 1.0)
            # self.paramSet[i] = self.mu + np.sqrt(self.cov) * noise
            # print(noise)
            # print(e)
            self.paramSet[i] = theta

    def updatePopParam(self):
        elite_idx = np.argsort(self.scoreSet)[-self.nElite:]
        scoreElite = self.scoreSet[elite_idx]
        self.best_idx = elite_idx[-1]
        #######################################################################
        # Task 3: Update distribution using samples
        # 1) Get elite parameters using "elite_idx", "self.paramSet"
        # 2) Compute the updated mean of distribution using elite parameters
        #    and assign it to "self.mu"
        #    (self.mu.shape = (self.paramSize, ))
        # 3) Compute the updated (diagonal) covariance of distribution
        #    using elite parameters and assign it to "self.cov"
        #    (self.cov.shape = (self.paramSize, ))
        #######################################################################
        # for e_id in elite_idx:
        mu = self.paramSet[elite_idx].mean(0)

        var_sum = np.zeros((self.paramSize, self.paramSize))
        for e_id in elite_idx:
            var_sum += np.matmul((self.paramSet[e_id] - self.mu).reshape(-1, 1), (self.paramSet[e_id] - self.mu).reshape(1, -1))

        var_sum = var_sum / self.nElite

        self.cov = np.diag(var_sum)
        # print(self.cov)
        self.mu = mu
        self.pnet.set_params(self.mu)
        return scoreElite.mean(), scoreElite.max(), scoreElite.min()


def run(AgentClass=AgentDDPG, mode='train'):
    env = ContinuousCartPoleEnv()
    agent = AgentClass(env)
    if mode == 'train':
        agent.runMany()
    elif mode == 'test':
        agent.load()
    agent.runTest()
    env.close()


if __name__ == '__main__':
    """
    1) Select the agent you want to train or test
    2) Select whether to train the agent
       or test the best agent you already trained
    """
    agentClass = AgentDDPG
    # agentClass = AgentCEM

    mode = "train"
    # mode = "test"

    run(agentClass, mode)
