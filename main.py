# -*- coding:utf-8 -*-
'''
作者：cy
日期：2025年11月04日
6.6 卷积神经网络（LeNet）
'''
import torch
from torch import nn
from torch.utils import data
import torchvision
from torchvision import transforms
# from torch.utils.data import TensorDataset, DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt


# 6.6.1 LeNet
"""
LeNet（LeNet‐5）由两个部分组成：
• 卷积编码器：由两个卷积层组成;
• 全连接层密集块：由三个全连接层组成。
"""
"""
nn.AvgPool1d：处理一维数据（时序信号）
nn.AvgPool2d：处理二维数据（图像、特征图）
nn.AvgPool3d：处理三维数据（视频、体积数据）
kernel_size=5: 卷积核大小：5×5的卷积核

输出高度 = (输入高度 + 2×padding - kernel_size) / stride + 1
输出宽度 = (输入宽度 + 2×padding - kernel_size) / stride + 1
"""
# 后面的形状是自己的猜测，通过后面验证猜测对不对
net = nn.Sequential(nn.Conv2d(1,6,kernel_size=5,padding=2),nn.Sigmoid(),   # [1,6,28+4-5+1,28+4-5+1]=[1,6,28,28]
                    nn.AvgPool2d(kernel_size=2,stride=2),  # [1,6,14,14]
                    nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),  # [1,16,10,10]
                    nn.AvgPool2d(kernel_size=2,stride=2),   # [1,16,5,5]
                    nn.Flatten(),   # [1,16*5*5]=[1,400]
                    nn.Linear(16*5*5,120),nn.Sigmoid(),   # [1,120]
                    nn.Linear(120,84),nn.Sigmoid(),    # [1,84]
                    nn.Linear(84,10))     # [1,10]

"""
在 PyTorch 中，4D 图像张量的维度通常遵循 (N, C, H, W) 的格式：
N (Batch Size): 批量大小，表示这批数据里有多少张图片。
C (Channels): 通道数。
H (Height): 图像高度。
W (Width): 图像宽度。
"""
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
"""
1. layer.__class__.__name__
layer：当前层的对象
.__class__：获取对象的类
.__name__：获取类的名称
作用：打印层的类型名称（如：'Conv2d', 'ReLU', 'Linear'）
2. 'output shape: \t'
输出固定的文字说明
\t：制表符，用于对齐输出
"""
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)


# 6.6.2 模型训练
# LeNet在Fashion‐MNIST数据集上的表现
def load_data_fashion_mnist(batch_size,resize=None):
    trans = [transforms.ToTensor()]
    print("type(trans) = ",type(trans))
    if resize:
        trans.insert(0,transforms.Resize(resize))
    # 前面几行是设置图像操作顺序
    # 下面一行进行图像处理
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="./data",train=True,transform=trans,download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data",train=False,transform=trans,download=True)
    return data.DataLoader(mnist_train, batch_size, shuffle=True), data.DataLoader(mnist_test, batch_size, shuffle=False)

batch_size = 64
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

# 计算预测正确的数量，训练和评估都会通道
def accuracy(y_hat,y):
    """
    y_hat.shape - 预测结果的形状
    len(y_hat.shape) > 1 - 检查是否是二维或更高维张量
    y_hat.shape[1] > 1 - 检查是否是多分类问题（类别数>1）
    y_hat.argmax(axis=1) 是获取每行最大值的索引位置。
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis = 1)
    cmp = y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())

# 评估函数
def evaluate_accuracy_gpu(net,data_iter,device=None):
    if isinstance(net,nn.Module):
        net.eval()  # 设置为评估模式
        """
        net.parameters() - 返回模型的所有参数（权重和偏置）
        iter(net.parameters()) - 创建参数的迭代器
        next(iter(net.parameters())) - 获取第一个参数
        .device - 获取该参数所在的设备（CPU 或 GPU）
        """
        if not device:
            print('device', device)
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = [0.0]*2
    with torch.no_grad():
        for X, y in data_iter:
            # 将数据移动到指定设备（CPU或GPU） 的标准做法
            # 检查X是否是列表类型，如果是列表，遍历列表中的每个元素并转移
            # 为什么列表和元组在深度学习中被认为是"复杂数据结构"？
            # 张量：形状统一，内存连续
            # 列表/元组：包含不同形状、不同大小的张量
            if isinstance(X,list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            """
            accuracy(net(X), y) - 计算这批数据的正确预测数量
            y.numel() - 获取这批数据的样本总数
            """
            # metric.add(accuracy(net(X),y), y.numel())
            metric = [a + float(b) for a, b in zip(metric, [accuracy(net(X),y), y.numel()])]
    return metric[0]/metric[1]

class Timer:
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()
    def start(self):
        """启动计时器"""
        # time.time()：返回当前时间的时间戳（从1970年1月1日开始的秒数，浮点数）
        self.tik = time.time()
    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time()-self.tik)
        return self.times[-1]
    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)
    def sum(self):
        """返回时间总和"""
        return sum(self.times)
    def cumsum(self):
        """返回累计时间"""
        """
        np.array(self.times) - 将列表转换为NumPy数组
        .cumsum() - 计算累积和
        .tolist() - 转换回Python列表
        """
        return np.array(self.times).cumsum().tolist()


def plot_training_curves(train_losses, train_accuracies, test_accuracies, num_epochs):
    """绘制训练曲线"""
    epochs = range(1, num_epochs + 1)

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 绘制损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 绘制准确率曲线
    ax2.plot(epochs, train_accuracies, 'r-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, test_accuracies, 'g--', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 在进行正向和反向传播之前，需要将每一小批量数据移动到我们指定的设备（例如GPU）上。
def train_ch6(net,train_iter,test_iter,num_epochs,lr,device):
    # 使用 Xavier均匀初始化 方法来初始化线性层和卷积层的权重
    """
    使用Xavier初始化的好处：
    保持激活值的方差在合理范围内
    避免梯度消失或爆炸
    加速模型收敛

    m 参数是通过 PyTorch 的 .apply() 方法 自动传递的：
    net.apply(init_weight)  # 这里会自动传递每个模块作为 m 参数
    """
    def init_weight(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weight)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(),lr=lr)
    loss = nn.CrossEntropyLoss()
    timer, num_batches = Timer(), len(train_iter)
    # 添加：用于存储历史数据的列表
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    # test_losses = []  # 一般不画测试损失，
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = [0.0] * 3
        for i, (X,y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X,y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat,y)
            l.backward()
            optimizer.step()  # 算法更新，这里可以使用自己设计的算法
            """
            with torch.no_grad(): 是为了防止 metric.add() 这一行中所有涉及张量的操作进行梯度计算。
            l * X.shape[0]还关联着之前的计算图
            乘法操作 l * X.shape[0] 会创建新的计算图节点
            没有 no_grad() 的话，这个乘法也会被记录用于梯度计算
            """
            with torch.no_grad():
                # 总的损失（不是平均），预测正确个数，样本总数
                #
                metric = [a + float(b) for a, b in zip(metric,[l*X.shape[0],accuracy(y_hat,y),X.shape[0]])]
            timer.stop()
            # 这里手动计算平均损失是因为在训练过程中，最后一个批次的样本数量可能不满
            train_1 = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            # // 向下取整
            # if (i+1)%(num_batches // 5) ==0 or i == num_batches -1:
        # 每个epoch中，全部训练完之后进行测试
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        # 存储历史数据
        train_losses.append(train_1)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        print(f'epoch: {epoch}, train mean accuracy: {train_acc: .3f}, test accuracy: {test_acc: .3f}')
    # 全部epoch结束后
    # metric[2] * num_epochs 表示所有epoch中使用的总样本数量。每个epoch使用的样本数量相同。
    plot_training_curves(train_losses, train_accuracies, test_accuracies, num_epochs)
    print(f'loss {train_1:.3f}, train acc {train_acc:.3f}, 'f'test acc {test_acc:.3f}')
    # print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec 'f'on {str(device)}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec 'f'on {device}')

def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        # f'' 表示 f-string（格式化字符串字面量）
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
# 训练和评估
lr, num_epochs = 0.9, 10
train_ch6(net,train_iter,test_iter,num_epochs,lr,try_gpu())
