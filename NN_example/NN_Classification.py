import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

class SLP(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super().__init__()
        self.fc = torch.nn.Linear(D_in, D_out)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        y = self.fc(x)
        return y  # softmax is done within F.cress_entropy
        # return F.softmax(y, dim=-1)

class TLP(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.fc1 = torch.nn.Linear(D_in, H)
        self.fc2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        h = self.fc1(x)
        h = F.relu(h)
        y = self.fc2(h)
        return y  # softmax is done within F.cross_entropy
        # return F.softmax(y, dim=-1)

class MLP(torch.nn.Module):
    def __init__(self, lsize):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_layers = len(lsize) - 1
        for i in range(self.n_layers):
            self.layers.append(torch.nn.Linear(lsize[i], lsize[i+1]))

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        for i in range(self.n_layers):
            x = self.layers[i](x)
            if i < self.n_layers-1:
                x = F.relu(x)
        return x  # softmax is done within F.cress_entropy
        # return F.softmax(x, dim=-1)


class CNN(nn.Module):
    def __init__(self, i_size, i_channels, D_out):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(i_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        i_size = i_size//4
        self.fc = nn.Linear(i_size*i_size*32, D_out)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out  # softmax is done within F.cress_entropy
        # return F.softmax(out, dim=-1)


def compute_accuracy(predictions, labels, cm=None):
    total = labels.size(0)
    _, predicted = torch.max(predictions.data, 1)
    correct = (predicted == labels).sum().item()
    acc = correct / total
    if cm is not None:
        for i in range(total):
            c = labels[i]
            p = predicted[i]
            cm[c][p] += 1
    return correct, acc


def test(test_loader, model):
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        cm = np.zeros((10,10), dtype="i")
        for images, labels in test_loader:
            images = images.to(model.device)
            labels = labels.to(model.device)
            outputs = model(images)
            nc, _ = compute_accuracy(outputs, labels, cm)
            total += labels.size(0)
            correct += nc
        print('Test Accuracy of the model on the 10000 test images: {:.2f}%'.format((correct / total) * 100))
        print(cm)


def train(train_loader, model, optimizer, acc_list):
    # Train the model
    total_step = len(train_loader)
    incorrect = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(model.device)
        labels = labels.to(model.device)
        m = len(labels)

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = F.cross_entropy(outputs, labels)


        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        n_correct, accuracy = compute_accuracy(outputs, labels)
        incorrect += m - n_correct
        acc_list.append(accuracy)

        if (i + 1) % 100 == 0:
            print('Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(i + 1, total_step, loss.item(), accuracy*100))

    print ("incorrect = ", incorrect)
    # incorrect = 0


def main_classify():
    stime = time.time()

    num_epochs = 20
    batch_size = 128
    MNIST = True

    if MNIST:
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(root='MNISTData', train=True, transform=trans, download=True)
        test_dataset = datasets.MNIST(root='MNISTData', train=False, transform=trans, download=True)
        input_channels = 1
        input_size = 28
    else:  # CIFAR10
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(root='CIFAR10Data', train=True, transform=trans, download=True)
        test_dataset = datasets.CIFAR10(root='CIFAR10Data', train=False, transform=trans, download=True)
        input_channels = 3
        input_size = 32

    n_in = input_size * input_size * input_channels  # for MNIST
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    n_out = 10

    # model = SLP(n_in,n_out)
    # model = TLP(n_in, 256, n_out)
    model = MLP([n_in, 256, 256, 256, n_out])
    # model = CNN(input_size, input_channels, n_out)

    # model.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.device = torch.device('cpu')
    model.to(model.device)

    learning_rate = 0.001
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(),
    #                   lr=learning_rate,
    #                   momentum=0.9,
    #                   nesterov=True,
    #                   weight_decay=0.0000
    # )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    acc_list = []
    for epoch in range(num_epochs):
        lr_scheduler.step()
        print ("Epoch %d / %d"%(epoch+1, num_epochs))
        train(train_loader, model, optimizer, acc_list)
        test(test_loader, model)

    # model.cpu()
    # show_all_filters(model)  # works only for CNN model

    etime = time.time()
    print ("Time : ", (etime-stime), " sec.")
    plt.plot(acc_list)
    plt.show()


if __name__ == '__main__':
    main_classify()
