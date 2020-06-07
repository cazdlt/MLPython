import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import fetch_openml
from torch.utils import data as data_utils
from torchvision import datasets, transforms
import numpy as np

# Haciendo esto acelerado para aprender algo de Pytorch

batch_size=64
print_idx=16
alpha=0.1
num_epochs=10
num_samples=6400

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
X_train=datasets.MNIST(root="./Data", train=True, transform=transform, download=True)
X_test=datasets.MNIST(root="./Data", train=False, transform=transform, download=True)

training_samples=range(num_samples)

trainloader=data_utils.DataLoader(X_train,batch_size=batch_size, sampler=data_utils.SubsetRandomSampler(training_samples))
testloader=data_utils.DataLoader(X_test,batch_size=batch_size, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(784, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):

        x = x.view(-1, 784)           
        x = F.leaky_relu(self.fc1(x))
        x=self.fc2(x)
        x = F.relu(x)
        return x

net = Net()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=alpha)

for epoch in range(num_epochs):
    train_loader_iter = iter(trainloader)
    cum_loss=0
    for batch_idx, (inputs, labels) in enumerate(train_loader_iter):
        net.zero_grad()
        inputs, labels = inputs, labels
        output = net(inputs)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()

        cum_loss+=loss
        if batch_idx % print_idx == 0 and batch_idx!=0:
            print(f"Iteration: {epoch+1} - {batch_idx}", f"Loss: {cum_loss/print_idx:.3f}")
            cum_loss=0

#test
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        #print(outputs)
        _, predicted = torch.max(outputs.data, 1)
        #print(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(total,correct)