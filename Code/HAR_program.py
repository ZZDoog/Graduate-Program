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
            label = torch.tensor([0])
        elif 'handclapping' in item_name:
            label = torch.tensor([1])
        elif 'handwaving' in item_name:
            label = torch.tensor([2])
        elif 'jogging' in item_name:
            label = torch.tensor([3])
        elif 'running' in item_name:
            label = torch.tensor([4])
        elif 'walking' in item_name:
            label = torch.tensor([5])

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
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1)
    print("train data load success!")
    print("loading test data.......")
    test_data = get_dataset(TEST_PATH)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)
    print("test data load success!")

    # create the LSTM network
    model = HARModel()
    model.cuda()

    # deefine loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    # start training the network
    for epoch in range(3000):
        running_loss = 0.0
        for i, data in enumerate(train_data, 0):

            inputs, labels = data
            inputs = torch.tensor(inputs)

            inputs = inputs.type(torch.FloatTensor)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            outputs = outputs.type(torch.FloatTensor)
            outputs = outputs.to(device)
            outputs = torch.unsqueeze(outputs, 0)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i+1) % 200 == 0:
                print('[%d , %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss / 200))
                with open('loss_record_2022.5.14.txt', 'a') as f:
                    f.write('%.5f ' % (running_loss / 200))
                running_loss = 0.0


        if (epoch+1)%5 == 0:
            test_num_total = 0
            test_num_correct = 0
            with torch.no_grad():
                for i_test, data_test in enumerate(test_data, 0):
                    input_test, label_test = data_test

                    input_test = torch.tensor(input_test)

                    input_test = input_test.type(torch.FloatTensor)
                    input_test, label_test = input_test.to(device), label_test.to(device)

                    output_test = model(input_test)
                    output_test = torch.squeeze(output_test)

                    output_test = output_test.type(torch.FloatTensor)
                    output_test = output_test.to(device)

                    if output_test.argmax(0) == label_test:
                        test_num_total += 1
                        test_num_correct += 1
                    else:
                        test_num_total += 1
            print('total num: %d, correct_num: %d' % (test_num_total, test_num_correct))
            print('Accuracy: %.3f %%' % (test_num_correct * 100 / test_num_total))
            with open('Accurcy_record_2022.5.14.txt', 'a') as f:
                f.write('%.3f ' % (test_num_correct*100 / test_num_total))

    loss_figure1 = np.loadtxt("loss_record_2022.5.14.txt")
    acc_figure1 = np.loadtxt("Accurcy_record_2022.5.14.txt")

    x1 = np.arange(1, 501, 1)
    y1 = loss_figure1
    x2 = np.arange(1, 101, 1)
    y2 = acc_figure1

    fig = plt.figure(1)
    ax1 = plt.subplot(1, 2, 1)
    plt.plot(x1, y1, color='r', label='CNN-LSTM')
    ax1.legend(loc="best", labelspacing=1, handlelength=2, fontsize=8, shadow=False)

    plt.xlabel('Training epoch')
    plt.ylabel('Training loss')
    ax2 = plt.subplot(1, 2, 2)
    plt.xlabel('Evaluate epoch')
    plt.ylabel('Evaluate Accuracy')
    plt.plot(x2, y2, color='b', label='CNN-LSTM')
    ax2.legend(loc="best", labelspacing=1, handlelength=2, fontsize=8, shadow=False)
    plt.show()





