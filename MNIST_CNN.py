#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 13:17:43 2018

@author: tpeng2
"""

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 10
num_classes = 10
batch_size = 60 
learning_rate = 3

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=5, stride=1, padding=0),#nx-kernel+1 
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  #24/2=12
        self.layer2 = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(4*4*16, num_classes)  #8/2=4, match last channel 16
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = F.dropout(out, training=self.training)
        out = F.log_softmax(out,dim=1)
        return out

model = ConvNet(num_classes).to(device)
NumParams=sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters in model =',NumParams)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
LossesToPlot=np.zeros(num_epochs*total_step)
j=0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        LossesToPlot[j]=loss.item()
        j+=1
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

print('Final Loss = ',loss.item()) # Print final loss

# Plot Loss as a function of Mini-Batch
plt.plot(LossesToPlot)
plt.xlabel('Mini-Batch number')
plt.ylabel('Loss')

#%% Training data
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predicted = torch.argmax(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Percent of training images misclassified: {} %'.format(100-100 * correct / total))
#%% Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
    print('Percent of testing images misclassified: {} %'.format(100-100 * correct / total))

#%% Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
wn0=np.reshape(model.fc.cpu().weight[0,].detach().numpy(),[16,16])
wn1=np.reshape(model.fc.cpu().weight[1,].detach().numpy(),[16,16])
wn2=np.reshape(model.fc.cpu().weight[2,].detach().numpy(),[16,16])
wn3=np.reshape(model.fc.cpu().weight[3,].detach().numpy(),[16,16])
wn4=np.reshape(model.fc.cpu().weight[4,].detach().numpy(),[16,16])
wn5=np.reshape(model.fc.cpu().weight[5,].detach().numpy(),[16,16])
wn6=np.reshape(model.fc.cpu().weight[6,].detach().numpy(),[16,16])
wn7=np.reshape(model.fc.cpu().weight[7,].detach().numpy(),[16,16])
wn8=np.reshape(model.fc.cpu().weight[8,].detach().numpy(),[16,16])
wn9=np.reshape(model.fc.cpu().weight[9,].detach().numpy(),[16,16])

#%% Plot
plt.figure();
plt.subplot(2,5,1);plt.imshow(np.reshape(wn0, [16,16]),cmap="Greys");plt.title('0');
plt.subplot(2,5,2);plt.imshow(np.reshape(wn1, [16,16]),cmap="Greys");plt.title('1');
plt.subplot(2,5,3);plt.imshow(np.reshape(wn2, [16,16]),cmap="Greys");plt.title('2');
plt.subplot(2,5,4);plt.imshow(np.reshape(wn3, [16,16]),cmap="Greys");plt.title('3');
plt.subplot(2,5,5);plt.imshow(np.reshape(wn4, [16,16]),cmap="Greys");plt.title('4');
plt.subplot(2,5,6);plt.imshow(np.reshape(wn5, [16,16]),cmap="Greys");plt.title('5');
plt.subplot(2,5,7);plt.imshow(np.reshape(wn6, [16,16]),cmap="Greys");plt.title('6');
plt.subplot(2,5,8);plt.imshow(np.reshape(wn7, [16,16]),cmap="Greys");plt.title('7');
plt.subplot(2,5,9);plt.imshow(np.reshape(wn8, [16,16]),cmap="Greys");plt.title('8');
plt.subplot(2,5,10);plt.imshow(np.reshape(wn9, [16,16]),cmap="Greys");plt.title('9');
plt.tight_layout()
plt.savefig('prob3_weight.eps')