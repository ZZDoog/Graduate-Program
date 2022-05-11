import os
import sys

import numpy
import numpy as np
import _pickle as cp
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import cv2 as cv
import math

import torch
from torch import nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

TRAIN_PATH = 'D:\Graduate Program\Train Data\KTH\Train_data'
TEST_PATH = 'D:\Graduate Program\Train Data\KTH\Test_data'
MODEL_PATH = 'D:\Graduate Program\Train Data\KTH\Model'

MAX_FRAMENUM = 50
VIDEO_WIDTH = 160
VIDEO_HEIGHT = 120

train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
    device = "cuda:0"
else:
    print('No GPU available, training on CPU.')
    device = "cpu"

def count_sample(path):
    count=0
    int(count)
    for file in os.listdir(path):  # file 表示的是文件名
        count = count + 1
    return count


def load_dataset(sample_num, path):

    files = os.listdir(path)

    y = np.zeros((sample_num, 1), dtype=np.float32)
    x = np.zeros((sample_num, MAX_FRAMENUM, 1, VIDEO_HEIGHT, VIDEO_WIDTH), dtype=np.float32) #存储视频用的numpy数组

    cnt_file = 0
    int(cnt_file)

    for file in files:
        videopath = path+"/"+file
        cap=cv.VideoCapture(videopath)

        cnt = 0
        while(cap.isOpened() and cnt<MAX_FRAMENUM):
            a, b = cap.read()
            if a ==False: break
            gray = cv.cvtColor(b, cv.COLOR_BGR2GRAY)/255    #将3通道的RGB图像转换为灰度图像
            x[cnt_file][cnt][0] = gray
            cnt += 1
            for i in range(3):
                a, b = cap.read()
                if a == False: break
        cap.release()


        if 'boxing' in file:
            y[cnt_file] = 0.0
        elif 'handclapping' in file:
            y[cnt_file] = 0.0
        elif 'handwaving' in file:
            y[cnt_file] = 0.0
        elif 'jogging' in file:
            y[cnt_file] = 1.0
        elif 'running' in file:
            y[cnt_file] = 1.0
        elif 'walking' in file:
            y[cnt_file] = 2.0

        cnt_file += 1
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # dataloader_x = torch.utils.data.DataLoader(x, batch_size=sample_num, shuffle=False, num_workers=2)

    return x, y


class HARModel(nn.Module):

    def __init__(self, n_layers=1, drop_prob=0.5):

        super(HARModel, self).__init__()
        self.drop_prob = drop_prob

        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 4), stride=(3, 4))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.lstm = nn.LSTM(input_size=400, hidden_size=100, num_layers=n_layers, batch_first=True)

        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 16)
        self.fc3 = nn.Linear(16, 1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):

        x = self.conv(x)
        x = self.pool(x)

        #batch_first=True时，lstm的imput格式为（batch_size,len,channels)
        x = x.view(1, MAX_FRAMENUM, 120*160)         #(batch_size, len ,channel)
        x, (h_s, h_c) = self.lstm(x)

        output = torch.sigmoid(self.fc1(h_c))
        output = torch.sigmoid(self.fc2(output))
        output = torch.relu(self.fc3(output))

        output = torch.squeeze(output)

        return output

class LSTM(nn.Module):

    def __init__(self, n_layers=1, drop_prob=0.5):

        super(LSTM, self).__init__()
        self.drop_prob = drop_prob

        self.lstm = nn.LSTM(input_size=120*160, hidden_size=100, num_layers=n_layers, batch_first=True)

        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 16)
        self.fc3 = nn.Linear(16, 1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):

        # x = self.conv(x)
        # x = self.pool(x)

        #batch_first=True时，lstm的imput格式为（batch_size,len,channels)
        x = x.view(1, MAX_FRAMENUM, 120*160)         #(batch_size, len ,channel)
        x, (h_s, h_c) = self.lstm(x)

        output = torch.sigmoid(self.fc1(h_c))
        output = torch.sigmoid(self.fc2(output))
        output = torch.relu(self.fc3(output))

        output = torch.squeeze(output)


        return output




if __name__ == '__main__':

    #load the train data and the test data
    print("loading train data.......")
    x_train, y_train = load_dataset(count_sample(TRAIN_PATH), TRAIN_PATH)
    print("train data load success!")
    print("loading test data.......")
    x_test,y_test = load_dataset(count_sample(TEST_PATH), TEST_PATH)
    print("test data load success!")

    #create the LSTM network
    model = HARModel()
    model.cuda()

    #print (x_train.shape)

    #deefine loss function
    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    #start training the network
    for epoch in range(500):
        running_loss = 0.0
        for i, data in enumerate(x_train):
            inputs = data
            labels = y_train[i]
            inputs, labels = torch.tensor(inputs), torch.tensor(labels)
            #inputs, labels = Variable(inputs), Variable(labels)
            #inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            inputs, labels = inputs.type(torch.FloatTensor), labels.type(torch.FloatTensor)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.type(torch.FloatTensor)

            outputs = torch.unsqueeze(outputs, 0)
            outputs = torch.unsqueeze(outputs, 0)
            labels = torch.unsqueeze(labels, 0)
            outputs = outputs.to(device)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print('[%d , %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss/200))
                with open('loss_record_nocnn_3class.txt', 'a') as f:
                    f.write('%.5f ' % (running_loss/200))
                running_loss = 0.0


        if (epoch+1) % 5 == 0:
            test_num_total = 0
            test_num_correct = 0
            for i, data in enumerate(x_test):
                inputs = data
                labels = y_test[i]

                inputs, labels = torch.tensor(inputs), torch.tensor(labels)
                inputs, labels = inputs.type(torch.FloatTensor), labels.type(torch.FloatTensor)
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                test_num_total += 1
                if math.fabs(outputs - labels) < 0.5:
                    test_num_correct += 1
                else:
                    print(labels)

            print('total num: %d, correct_num: %d' % (test_num_total, test_num_correct))
            print('Accuracy: %.3f %%' % (test_num_correct*100 / test_num_total))

            with open('Accurcy_record_nocnn_3class.txt', 'a') as f:
                f.write('%.3f ' % (test_num_correct*100 / test_num_total))

    loss_figure1 = np.loadtxt("loss_record_3.txt")
    acc_figure1 = np.loadtxt("Accurcy_record_3.txt")
    loss_figure2 = np.loadtxt("loss_record_nocnn_3class.txt")
    acc_figure2 = np.loadtxt("Accurcy_record_nocnn_3class.txt")

    x1 = np.arange(1, 501, 1)
    y1 = loss_figure1
    y2 = loss_figure2
    x2 = np.arange(1, 101, 1)
    y3 = acc_figure1
    y4 = acc_figure2

    fig = plt.figure(1)
    ax1 = plt.subplot(1, 2, 1)
    plt.plot(x1, y1, color='r', label='CNN-LSTM')
    plt.plot(x1, y2, color='g', label='regular LSTM')
    ax1.legend(loc="best", labelspacing=1, handlelength=2, fontsize=8, shadow=False)

    plt.xlabel('Training epoch')
    plt.ylabel('Training loss')
    ax2 = plt.subplot(1, 2, 2)
    plt.xlabel('Evaluate epoch')
    plt.ylabel('Evaluate Accuracy')
    plt.plot(x2, y3, color='b', label='CNN-LSTM')
    plt.plot(x2, y4, color='y', label='regular LSTM')
    ax2.legend(loc="best", labelspacing=1, handlelength=2, fontsize=8, shadow=False)
    plt.show()







