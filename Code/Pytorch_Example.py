# -*- coding:utf-8 -*-
import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
#数据预处理：转换为Tensor，归一化，设置训练集和验证集以及加载子进程数目
transform = transforms.Compose([transforms.ToTensor() , transforms.Normalize((0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5))])  #前面参数是均值，后面是标准差
trainset = torchvision.datasets.CIFAR10(root = './data' , train = True , download = True , transform = transform)
trainloader = torch.utils.data.DataLoader(trainset , batch_size = 4 , shuffle = True , num_workers =2)  #num_works = 2表示使用两个子进程加载数据
testset = torchvision.datasets.CIFAR10(root = './data' , train = False , download = True , transform = transform)
testloader = torch.utils.data.DataLoader(testset , batch_size = 4 , shuffle = True , num_workers = 2)
classes = ('plane' , 'car' , 'bird' , 'cat' , 'deer' , 'dog' , 'frog' , 'horse' , 'ship' , 'truck')


import matplotlib.pyplot as plt
import numpy as np
import pylab

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg , (1 , 2 , 0)))
    pylab.show()

dataiter = iter(trainloader)
images , labels = dataiter.next()
for i in range(4):
    p = plt.subplot()
    p.set_title("label: %5s" % classes[labels[i]])
    imshow(images[i])


#构建网络
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net , self).__init__()
        self.conv1 = nn.Conv2d(3 , 6 , 5)
        self.pool = nn.MaxPool2d(2 , 2)
        self.conv2 = nn.Conv2d(6 , 16 , 5)
        self.fc1 = nn.Linear(16 * 5 * 5 , 120)
        self.fc2 = nn.Linear(120 , 84)
        self.fc3 = nn.Linear(84 , 10)

    def forward(self , x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1 , 16 * 5 * 5)  #利用view函数使得conv2层输出的16*5*5维的特征图尺寸变为400大小从而方便后面的全连接层的连接
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
net.cuda()

#define loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters() , lr = 0.001 , momentum = 0.9)

#train the Network
for epoch in range(2):
    running_loss = 0.0
    for i , data in enumerate(trainloader , 0):
        inputs , labels = data
        inputs , labels = Variable(inputs.cuda()) , Variable(labels.cuda())
        optimizer.zero_grad()
        #forward + backward + optimizer
        outputs = net(inputs)
        loss = criterion(outputs , labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if i % 2000 == 1999:
            print('[%d , %5d] loss: %.3f' % (epoch + 1 , i + 1 , running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

dataiter = iter(testloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print('GroundTruth:' , ' '.join(classes[labels[j]] for j in range(4)))

outputs = net(Variable(images.cuda()))

_ , predicted = torch.max(outputs.data , 1)
print('Predicted: ' , ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

correct = 0
total = 0
for data in testloader:
    images , labels = data
    outputs = net(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data , 1)
    correct += (predicted == labels.cuda()).sum()
    total += labels.size(0)
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

class_correct = torch.ones(10).cuda()
class_total = torch.ones(10).cuda()
for data in testloader:
    images , labels = data
    outputs = net(Variable(images.cuda()))
    _ , predicted = torch.max(outputs.data , 1)
    c = (predicted == labels.cuda()).squeeze()
    #print(predicted.data[0])
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i] , 100 * class_correct[i] / class_total[i]))