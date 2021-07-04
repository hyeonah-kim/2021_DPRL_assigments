import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

class MLP(torch.nn.Module):
    def __init__(self, lsize):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_layers = len(lsize) - 1
        # self.lrelu = torch.nn.LeakyReLU(0.01)
        for i in range(self.n_layers):
            self.layers.append(torch.nn.Linear(lsize[i], lsize[i+1]))

        # self.hmean = torch.zeros(lsize[1])
        # self.hvar = torch.zeros(lsize[1])

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        for i in range(self.n_layers):
            x = self.layers[i](x)
            if i < self.n_layers-1:
                x = F.relu(x)
                # x = self.lrelu(x)
                # with torch.no_grad():
                #     self.hmean += x.mean(dim=0)
                #     self.hvar += x.var(dim=0)
        return x


def train_reg(Xt, Yt, model, opt, loss_list):
    N = len(Xt)
    m = 512  # minibatch size
    n_batch = N // m
    for i in range(n_batch):
        idx = np.random.choice(N, m, replace=False)  # 0 ~ N, m개 비복원추출
        mb_X = Xt[idx,0].view([m,1])
        mb_Y = Yt[idx,0].view([m,1])
        # Forward pass
        Y_hat = model(mb_X)
        loss = F.mse_loss(mb_Y, Y_hat)
        loss_list.append(loss.item())

        # Backprop
        opt.zero_grad()
        loss.backward()
        opt.step()

        print('Step [{}/{}], Loss: {:.4f}'.format(i+1, n_batch, loss.item()))


def generate_data(n, sigma=0.02):
    X = np.random.rand(n,1)*4
    Y = 0.5*X + 0.5
    for i in range(n):
        x = X[i,0]
        if x > 1:
            if x < 2:
                Y[i,0] = 0.5*(x-2)**2 + 0.5
            else:
                Y[i,0] = np.log(x - 1) + 0.5
    Y += np.random.randn(n,1)*sigma
    return X,Y

def main_reg():
    stime = time.time()
    H = 8
    model = MLP([1,H,1])
    loss_list = []

    N = 5000
    X, Y = generate_data(N)
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)

    N_test = 2000
    X_test, Y_test = generate_data(N_test)
    Xt_test = torch.tensor(X_test, dtype=torch.float32).view((N_test, 1))  # view[n, m] reshape as a n x m tensor

    plt.ion()
    plt.plot(X, Y, 'b.')
    plt.draw()
    plt.pause(1)

    num_epoch = 50
    learning_rate = 0.01
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.000)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for i in range(num_epoch):
        lr_scheduler.step()
        print("Epoch = ", i)
        train_reg(Xt, Yt, model, optimizer, loss_list)
        if i % 1 == 0:
            with torch.no_grad():
                Yhat_test = model(Xt_test)

            plt.gcf().clear()
            plt.plot(X, Y, 'b.')
            plt.plot(X_test, Yhat_test.numpy(), 'r.')
            plt.draw()
            plt.pause(0.1)

    plt.ioff()
    plt.figure()
    plt.plot(loss_list)
    plt.show()
    etime = time.time()
    print ("time = ", etime - stime)

if __name__ == '__main__':
    # main_MNIST()
    main_reg()
