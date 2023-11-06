# 导入所需的库和模块
import argparse
import os
import pickle
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import initializers
from keras import regularizers
from keras.layers import Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import Dense

# my_utils 是一个自定义的工具模块，用于路径操作
from my_utils import utils_paths

# 设置输入参数
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="图像数据集的路径")
ap.add_argument("-m", "--model", required=True, help="训练完成的模型保存路径")
ap.add_argument("-l", "--label-bin", required=True, help="标签二值化器的保存路径")
ap.add_argument("-p", "--plot", required=True, help="准确率/损失图的保存路径")
args = vars(ap.parse_args())

# 读取图像数据并打乱顺序
print("[INFO] 开始读取数据...")
data = []
labels = []
imagePaths = sorted(list(utils_paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# 逐一读取图像和标签
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32)).flatten()  # 将图像调整为32x32大小并转换成一维数组
    data.append(image)

    # 从文件路径中提取标签
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# 将图像数据缩放到[0, 1]区间
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# 划分训练集和测试集
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# 将标签转换为one-hot编码形式
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# 构建网络模型: 输入层3072个节点，两个隐藏层分别为512和256个节点，输出层节点数为类别数
model = Sequential()
model.add(Dense(512, input_shape=(3072,), activation="relu",
                kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.05),
                kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))  # 使用Dropout防止过拟合
model.add(Dense(256, activation="relu",
                kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.05),
                kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(len(lb.classes_), activation="softmax",
                kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.05),
                kernel_regularizer=regularizers.l2(0.01)))

# 设置优化器和编译模型
INIT_LR = 0.001  # 初始学习率
EPOCHS = 75  # 迭代次数
print("[INFO] 准备训练网络...")
opt = SGD(learning_rate=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# 开始训练模型
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32)

# 评估模型性能
print("[INFO] 正在评估模型...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

# 绘制训练过程中的损失和准确率曲线
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])

# 保存模型和标签二值化器到指定路径
print("[INFO] 正在保存模型...")
model.save(args["model"])
with open(args["label-bin"], "wb") as f:
    f.write(pickle.dumps(lb))
