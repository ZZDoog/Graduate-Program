import numpy as np
import matplotlib.pyplot as plt

loss_figure1 = np.loadtxt("loss_record_6class.txt")
acc_figure1 = np.loadtxt("Accurcy_record_6class.txt")
loss_figure2 = np.loadtxt("loss_record_nocnn_6class.txt")
acc_figure2 = np.loadtxt("Accurcy_record_nocnn_6class.txt")

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