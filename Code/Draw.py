import os
import numpy as np
import matplotlib.pyplot as plt

train_loss = np.loadtxt("./draw_data/train_loss_record_2022.5.23.txt")
train_acc = np.loadtxt("./draw_data/train_Accurcy_record_2022.5.23.txt")
test_loss = np.loadtxt("./draw_data/test_loss_record_2022.5.23.txt")
test_acc = np.loadtxt("./draw_data/test_Accurcy_record_2022.5.23.txt")


train_x = np.arange(1, 501, 1)
y1 = train_loss
y2 = test_loss
test_x = np.arange(1, 101, 1)
y3 = train_acc
y4 = test_acc

fig = plt.figure(1)

ax1 = plt.subplot(2, 2, 1)
plt.xlabel('Training epoch')
plt.ylabel('Training loss')
plt.plot(train_x, y1, color='r', label='CNN-LSTM:train loss')
ax1.legend(loc="best", labelspacing=1, handlelength=2, fontsize=8, shadow=False)


ax2 = plt.subplot(2, 2, 2)
plt.xlabel('Training epoch')
plt.ylabel('Training acc')
plt.plot(train_x, y3, color='r', label='CNN-LSTM:train acc')
ax2.legend(loc="best", labelspacing=1, handlelength=2, fontsize=8, shadow=False)

ax2 = plt.subplot(2, 2, 3)
plt.xlabel('Evaluate epoch')
plt.ylabel('Evaluate loss')
plt.plot(test_x, y2, color='b', label='CNN-LSTM:test loss')
ax2.legend(loc="best", labelspacing=1, handlelength=2, fontsize=8, shadow=False)

ax2 = plt.subplot(2, 2, 4)
plt.xlabel('Evaluate epoch')
plt.ylabel('Evaluate Accuracy')
plt.plot(test_x, y4, color='b', label='CNN-LSTM:test acc')
ax2.legend(loc="best", labelspacing=1, handlelength=2, fontsize=8, shadow=False)


plt.show()