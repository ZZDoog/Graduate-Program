import os

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset


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

class get_dataset(Dataset):
    def __init__(self, root_path):
        # self
        self.root_path = root_path
        self.item_list = os.listdir(self.root_path)

    def __getitem__(self, idx):
        item_name = self.item_list[idx]
        item_path = self.root_path + "/" + item_name

        item_video = np.zeros((MAX_FRAMENUM, 1, VIDEO_HEIGHT, VIDEO_WIDTH), dtype=np.float32) #存储视频用的numpy数组

        cap = cv.VideoCapture(item_path)
        cnt = 0
        while (cap.isOpened() and cnt < MAX_FRAMENUM):
            a, b = cap.read()
            if a == False: break
            gray = cv.cvtColor(b, cv.COLOR_BGR2GRAY) / 255  # 将3通道的RGB图像转换为灰度图像
            item_video[cnt][0] = gray
            cnt += 1
            for i in range(3):
                a, b = cap.read()
                if a == False: break

        if 'boxing' in item_name:
            label = 'boxing'
        elif 'handclapping' in item_name:
            label = 'hand_clapping'
        elif 'handwaving' in item_name:
            label = 'handwaving'
        elif 'jogging' in item_name:
            label = 'jogging'
        elif 'running' in item_name:
            label = 'running'
        elif 'walking' in item_name:
            label = 'walking'

        return item_video, label

    def __len__(self):
        return len(self.item_list)


class HARModel(nn.Module):

    def __init__(self, n_layers=1, drop_prob=0.5):

        super(HARModel, self).__init__()
        self.drop_prob = drop_prob

        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 4), stride=(3, 4))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.lstm = nn.LSTM(input_size=400, hidden_size=100, num_layers=n_layers, batch_first=True)

        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 16)
        self.fc3 = nn.Linear(16, 6)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):

        x = self.conv(x)
        x = self.pool(x)

        #batch_first=True时，lstm的imput格式为（batch_size,len,channels)
        x = x.view(1, MAX_FRAMENUM, 400)         #(batch_size, len ,channel)
        x, (h_s, h_c) = self.lstm(x)

        output = torch.sigmoid(self.fc1(h_c))
        output = torch.sigmoid(self.fc2(output))
        output = self.fc3(output)

        output = torch.squeeze(output)

        return output

if __name__ == '__main__':

    # load the train data and the test data
    print("loading train data.......")
    train_data = get_dataset(TRAIN_PATH)
    print("train data load success!")
    print("loading test data.......")
    test_data = get_dataset(TEST_PATH)
    print("test data load success!")

    # create the LSTM network
    model = HARModel()
    model.cuda()

    # deefine loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    # start training the network
    for epoch in range(500):
        running_loss = 0.0
        item_num = 0
        for i, data in enumerate(train_data, 0):
            item_num = item_num + 1

            inputs, labels = data
            inputs = torch.tensor(inputs)

            inputs= inputs.type(torch.FloatTensor)
            inputs= inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            outputs = outputs.type(torch.FloatTensor)
            outputs = outputs.to(device)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if item_num % 200 == 199:
                print('[%d , %5d] loss: %.5f' % (epoch + 1, item_num + 1, running_loss / 200))
                with open('loss_record_nocnn_3class.txt', 'a') as f:
                    f.write('%.5f ' % (running_loss / 200))
                running_loss = 0.0