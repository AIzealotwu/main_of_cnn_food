import os
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torch.optim import SGD #优化器可做更改 这里保存SGD类型
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
# 若要使用训练好的模型预测测试集，则定义其转换器以及写入文件的函数。

# 以下函数用于读取图片格式数据
def readfile(path, label):
    # label 是一个布尔变量，代表是否需要回传y，对于测试集是不需要的
    # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
    image_dir = sorted(os.listdir((path))) 
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8) 
    # print(x)
    y = np.zeros((len(image_dir)), dtype=np.uint8) # label
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file)) # 读取图片
        x[i, :, :, :] = cv2.resize(img, (128,128)) # 将图片进行缩放，压缩为128*128*3
        if label:
            y[i] = int(file.split("_")[0]) # 前面是类别，后面是编号

    if label:
        return x, y
    else:
        return x


# 组合读取路径
workspace_dir = './data'
print("The system is reading the data--------->")
# print(os.path.join(workspace_dir, "training"))
train_x, train_y = readfile(os.path.join(workspace_dir, "train"), True)
val_x, val_y = readfile(os.path.join(workspace_dir, "val"), True)


# 图像预处理。用Compose把多个步骤整合
train_transform = transforms.Compose([
    transforms.ToPILImage(), # 将tensor转成PIL的格式
    transforms.RandomHorizontalFlip(), # 随机翻转
    transforms.RandomRotation(15), # 随机旋转
    transforms.ToTensor() # 转成 Tensor，并把数值normalization到[0,1]
])

# 写数据处理的DataLoader的类
class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    # 当我们集成了一个 Dataset类之后，我们需要重写 len 方法，该方法提供了dataset的大小；
    # getitem 方法， 该方法支持从 0 到 len(self)的索引
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X
batch_size = 4 # pytorch中dataloader的大小将根据batch_size的大小自动调整。
train_set = ImgDataset(train_x, train_y, train_transform) # 将数据包装成Dataset类
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set , batch_size= batch_size, shuffle=True) # 将数据处理成DataLoader
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

## Model
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), # 卷积
            nn.BatchNorm2d(64), # 归一化
            nn.ReLU(), # 激活函数
            nn.MaxPool2d(2,2,0), # 池化
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
        )
        # 定义全连接神经网络
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

## Training
# model to GPU
model = Classifier().cuda() 
loss = nn.CrossEntropyLoss()  # loss 使用 CrossEntropyLoss
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, dampening=0, nesterov=True) 
num_epoch = 10
epoch_l = []
acc_train_l = []
loss_train_l = []
acc_val_l = []
loss_val_l = []
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train() # 确保 model 是在 train model （开启 Dropout 等）
    for i, data in enumerate(train_loader):
        optimizer.zero_grad() # 用 optimizer 将 model 参数的 gradient 归零，准备下一次的更新
        train_pred = model(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward() # 利用 back propagation 算出每个参数的 gradient
        optimizer.step() #  optimizer 用 gradient 更新参数值, 所以step()要放在后面


        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1)==data[1].numpy())
        train_loss += batch_loss.item()


    model.eval() # 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化，测试阶段往往是单个图像的输入，不存在mini-batch的概念。所以将model改为eval模式后，BN的参数固定，并采用之前训练好的全局的mean和std；
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        epoch_l.append(epoch + 1)
        acc_train_l.append(train_acc / train_set.__len__())
        loss_train_l.append(train_loss / train_set.__len__())
        acc_val_l.append(val_acc / val_set.__len__())
        loss_val_l.append(val_loss / val_set.__len__())

# 下面开始绘图
plt.plot(epoch_l, acc_train_l, c='blue', marker='o', linestyle=':', label='train_acc')
plt.plot(epoch_l, acc_val_l, c='red', marker='*', linestyle='-', label='validation_acc')
plt.plot(epoch_l, loss_train_l, c='green', marker='+', linestyle='--', label='train_loss')
plt.plot(epoch_l, loss_val_l, c='yellow', marker='^', linestyle='--', label='validation_loss')
plt.legend()
plt.xlabel(u'epoch_num')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()