import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam


def angleReward(obs, r):
    # return r  # plain reward
    return np.float32(r + 10*np.abs(obs[2]))  # angle reward


def plotReturn(retHist, m=0):
    plt.plot(retHist)
    if m > 1 and m < len(retHist):
        cumsum = [0]
        movAvg = []
        for i, x in enumerate(retHist, 1):
            cumsum.append(cumsum[i-1] + x)
            if i < m:
                i0 = 0
                n = i
            else:
                i0 = i - m
                n = m
            ma = (cumsum[i] - cumsum[i0]) / n
            movAvg.append(ma)
        plt.plot(movAvg)
    plt.show()


class ReplayBuffer(object):
    def __init__(self, maxlen, dimState):
        self.maxlen = maxlen
        self.dimState = dimState
        self.state_buff = np.zeros((self.maxlen, self.dimState), dtype=np.float32)
        self.act_buff = np.zeros((self.maxlen, 1), dtype=np.int64)
        self.rew_buff = np.zeros(self.maxlen, dtype=np.float32)
        self.next_state_buff = np.zeros((self.maxlen, self.dimState), dtype=np.float32)
        self.mask_buff = np.zeros(self.maxlen, dtype=np.uint8)

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
        a = torch.LongTensor(self.act_buff[idx])
        r = torch.FloatTensor(self.rew_buff[idx])
        sp = torch.FloatTensor(self.next_state_buff[idx])
        mask = torch.Tensor(self.mask_buff[idx])
        return s, a, r, sp, mask

    def __len__(self):
        return self.filled

    def clear(self):
        self.state_buff = np.zeros((self.maxlen, self.dimState), dtype=np.float32)
        self.act_buff = np.zeros((self.maxlen, 1), dtype=np.int64)
        self.rew_buff = np.zeros(self.maxlen, dtype=np.float32)
        self.next_state_buff = np.zeros((self.maxlen, self.dimState), dtype=np.float32)
        self.mask_buff = np.zeros(self.maxlen, dtype=np.uint8)

        self.filled_i = 0
        self.curr_i = 0


class MLPQNet(nn.Module):
    def __init__(self, dimState, dimAction, hidden_dim):
        super(MLPQNet, self).__init__()

        self.fc1 = nn.Linear(dimState, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, dimAction)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        out = self.fc3(h)

        return out


class DQNAgent(object):
    def __init__(self, env, hidden_dim=128,
                 batch_size=64):

        # Environment information
        self.env = env
        self.dimState = env.observation_space.shape[0]
        self.dimAction = env.action_space.n
        self.gamma = 0.99  # discount factor

        # Q network initialization
        self.hidden_dim = hidden_dim
        self.qnet = MLPQNet(self.dimState, self.dimAction, hidden_dim)
        self.target_qnet = MLPQNet(self.dimState, self.dimAction, hidden_dim)
        # self.qnet = DuelingQNet(self.dimState, self.dimAction, hidden_dim)
        # self.target_qnet = DuelingQNet(self.dimState, self.dimAction, hidden_dim)
        self.hard_update()

        # Experience replay buffer
        self.memory = ReplayBuffer(100000, self.dimState)
        self.batch_size = batch_size

        # Learning parameters
        self.tau = 1e-2  # target network update rate
        self.lr = 5e-4  # learning rate
        self.optimizer = Adam(self.qnet.parameters(), lr=self.lr)

        # epsilon annealing parameters
        self.eps = 1.0
        self.eps_end = 0.01
        self.eps_step = 0.99

    def save(self):
        torch.save(self.qnet.state_dict(), "saved_model.pt")

    def load(self):
        self.qnet.load_state_dict(torch.load("saved_model.pt"))

    # Epsilon-greedy policy
    def getAction(self, state):
        p = np.random.random()
        if p < self.eps:
            return np.random.randint(self.dimAction)
        else:
            return self.qnet(state).argmax().item()

    def hard_update(self):
        for target, source in zip(self.target_qnet.parameters(),
                                  self.qnet.parameters()):
            target.data.copy_(source.data)

    def soft_update(self):
        for target, source in zip(self.target_qnet.parameters(),
                                  self.qnet.parameters()):
            average = target.data*(1. - self.tau) + source.data*self.tau
            target.data.copy_(average)

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        self.optimizer.zero_grad()
        ################################################
        # Task 1: fill your code here
        # Option 1: Implement 1-step DQN update
        # 1) Sample a mini-batch (s, a, r, sp, mask) from "self.memory"
        # 2) Calculate Q(s, a) using "self.qnet" with sampled (s, a)
        #    and assign it to "pred"
        # 3) Calculate TD target using "self.target_qnet"
        #    with sampled (r, sp, mask) and assign it to "target"

        # Option 2: Implement 1-step Double DQN update
        # 1) Sample a mini-batch (s, a, r, sp, mask) from "self.memory"
        # 2) Calculate Q(s, a) using "self.qnet" with sampled (s, a)
        #    and assign it to "pred"
        # 3) Calculate TD target using both "self.qnet" and "self.target_qnet"
        #    with sampled (r, sp, mask) and assign it to "target"
        ################################################
        s, a, r, sp, mask = self.memory.sample(self.batch_size)
        pred_q = self.qnet.forward(s)  # in: state / out: action => batch size x dimAction
        pred = pred_q.gather(1, a).squeeze()  # torch.Size([64, 1]) -> torch.Size([64])
        # self.target_qnet(sp).max(1) => values tensor, indices tensor
        #################### DQN #######################
        target = r + self.gamma * self.target_qnet(sp).max(1)[0] * (1 - mask)  # torch.Size([64])
        ################################################
        ################# Double DQN ###################
        # ap = self.qnet(sp).max(1)[1].unsqueeze(1)
        # target = r + self.gamma * self.target_qnet(sp).gather(1, ap).squeeze() * (1 - mask)
        ################################################
        loss = F.mse_loss(pred, target)
        loss.backward()
        clip_grad_norm_(self.qnet.parameters(), 2.)
        self.optimizer.step()

    # Simulate 1 episode using current Q network
    # and return a list of rewards
    def runEpisode(self, maxStep, render=False, test=False):
        s = self.env.reset()
        done = False
        rewards = []
        for et_i in range(maxStep):
            if render:
                self.env.render()
            s_tensor = torch.FloatTensor(s)
            if test:
                a = self.qnet(s_tensor).argmax().item()
            else:
                a = self.getAction(s_tensor)
            sp, r, done, _ = self.env.step(a)
            r = angleReward(sp, r)
            rewards.append(r)
            if not test:
                self.memory.push(s, a, r, sp, done)
                self.train()
                self.soft_update()
            s = sp
            if done:
                break

        return rewards

    # Test the current Q network by using greedy policy
    # and return a sum of rewards
    def runTest(self, maxStep=1000, render=True):
        rewards = self.runEpisode(maxStep=maxStep, render=render,
                                  test=True)
        ret = sum(rewards)
        nStep = len(rewards)
        print(f"Test episode, return = {ret:.1f} in {nStep} steps")
        return ret

    # Run multiple episodes to train the agent
    # and give a learning plot
    def runMany(self, nEpisode, maxStep):
        retHist = []
        testHist = []
        max5 = 0
        for ep_i in range(nEpisode):
            rewards = self.runEpisode(maxStep=maxStep)
            self.eps = max(self.eps_end, self.eps*self.eps_step)
            ret = sum(rewards)
            print(f"Train episode i={ep_i}, return = {ret:.1f}, eps = {self.eps:.2f}")
            retHist.append(ret)

            if ep_i > 4:
                avg5 = sum(retHist[-5:])/5  # average return of the last 5 episodes
                if avg5 > max5 and max(retHist[-5:]) == retHist[-1]:
                    max5 = avg5
                    print(f"iter {ep_i}, avg5 = {avg5:.1f} updated")
                    if avg5 > 100:
                        self.save()
                    if avg5 == maxStep:
                        print("Average reached {maxStep} after {ep_i} episodes. Solution found")
                        break
            test_ret = self.runTest(render=False)
            testHist.append(test_ret)
        plotReturn(testHist, 10)


class DuelingQNet(nn.Module):
    def __init__(self, dimState, dimAction, hidden_dim):
        super(DuelingQNet, self).__init__()

        self.fc1 = nn.Linear(dimState, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        ################################################
        # Task 2-1: fill your code here
        # Build the network of dueling part to calculate
        # V(s) and A(s, a) separately
        ################################################
        self.fc_v = nn.Linear(hidden_dim, 1)
        self.fc_a = nn.Linear(hidden_dim, dimAction)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        ################################################
        # Task 2-2: fill your code here
        # Implement the forward pass of the dueling part
        # and assign Q values to "out"
        ################################################
        v = self.fc_v(h)  # batch_size x 1
        a = self.fc_a(h)  # batch_size x 2
        # out = v + (a - torch.max(a, dim=-1, keepdim=True)[0])  # option1
        out = v + (a - (1/2) * torch.sum(a, dim=-1, keepdim=True))  # option2
        return out


def trainAgentDQN():
    env = gym.make('CartPole-v1').unwrapped
    agent = DQNAgent(env)
    agent.runMany(1000, maxStep=2000)
    env.close()


def testBestAgentDQN():
    env = gym.make('CartPole-v1').unwrapped
    agent = DQNAgent(env)
    agent.load()
    agent.runTest()
    env.close()


if __name__ == "__main__":
    trainAgentDQN()
    # testBestAgentDQN()
