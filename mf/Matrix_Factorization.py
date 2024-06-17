"""
想用pytorch实现下 Matrix Factorization,PyTorch（用于构建和训练神经网络）、Pandas（用于数据处理）、Scikit-learn（用于数据划分和评估指标）等。
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
path = './'
batch_size = 1024
device = torch.device('cuda:0')
learning_rate = 5e-4
weight_decay = 1e-5
epochs = 100

#创建一个名为MfDataset的类，用于封装推荐系统中的用户ID、商品ID和评分数据。这个类告诉PyTorch如何处理和加载数据。
class MfDataset(Dataset):
    def __init__(self, u_id, i_id, rating):
        self.u_id = u_id
        self.i_id = i_id
        self.rating = rating

    def __getitem__(self, index):
        return self.u_id[index], self.i_id[index], self.rating[index]

    def __len__(self):
        return len(self.rating)


class MF(nn.Module):
    def __init__(self, num_users, num_items, mean, embedding_size=100):#MF 是类名，它代表矩阵分解模型。是用户的数量。商品的数量。是所有评分的平均值，用作模型的全局偏置。是嵌入向量的维度，默认为 100。
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)#nn.Embedding 是 PyTorch 中用于创建嵌入层的类，它将离散的 ID 映射到连续的向量空间中。
        self.user_bias = nn.Embedding(num_users, 1)#偏置层偏置参数通常具有较小的数值，它们可以帮助模型更快地收敛，因为它们为模型提供了一个良好的初始起点。
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)
#weight.data.uniform_ 方法用于初始化权重，这里使用均匀分布。
#用户和商品嵌入层的权重被初始化为 [0, 0.005] 范围内的随机值，这有助于打破对称性。
#偏置层的权重被初始化为 [-0.01, 0.01] 范围内的随机值。
        self.user_emb.weight.data.uniform_(0, 0.005)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(0, 0.005)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        # 全局bias使用 nn.Parameter 创建一个参数，这里将评分的平均值作为全局偏置。
        # torch.FloatTensor([mean]) 创建一个包含平均值的浮点张量。
        # 第二个参数 False 表示这个参数在训练过程中不会被更新
        self.mean = nn.Parameter(torch.FloatTensor([mean]), False)

    #    方法定义了模型如何根据输入数据进行前向传播。
    #    U 和 I 是对应的嵌入向量。
    # b_u 和 b_i 是对应的偏置值，squeeze 方法用于去除单维度。
    def forward(self, u_id, i_id):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()

        return (U * I).sum(1) + b_u + b_i + self.mean
#U * I 计算用户嵌入和商品嵌入的逐元素乘积。.sum(1) 对逐元素乘积的结果进行求和，得到一个标量值，表示没有偏置的预测评分。
#b_u + b_i 分别加上用户和商品的偏置。
#+ self.mean 最后加上全局偏置，得到最终的预测评分。


def main():
    df = pd.read_csv(path + 'u.data', header=None, delimiter='\t')
    #将数据分为用户和商品  ID(x) 以及对应的评分(y)。
    x, y = df.iloc[:, :2], df.iloc[:, 2]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2020)
   # 使用  train_test_split函数将数据集划分为训练集和测试集，其中30 % 的数据用作测试。
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    # 存储训练损失和验证MSE
    train_losses, val_mses = [], []
    # 需要将数据全部转化为np.array, 否则后面的dataloader会报错， pytorch与numpy之间转换较好，与pandas转化容易出错
    train_dataset = MfDataset(np.array(x_train[0]), np.array(x_train[1]), np.array(y_train).astype(np.float32)) # 将标签设为np.float32类型， 否则会报错
    test_dataset = MfDataset(np.array(x_test[0]), np.array(x_test[1]), np.array(y_test).astype(np.float32))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)# DataLoader 对象，用于训练数据集 train_dataset。batch_size 是每个批次中的样本数量。DataLoader 会处理数据的批量加载，使得在训练模型时可以一次处理一批数据，而不是单个样本。
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


    mean_rating = df.iloc[:, 2].mean()
    num_users, num_items = max(df[0]) + 1, max(df[1]) + 1
    model = MF(num_users, num_items, mean_rating).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_func = torch.nn.MSELoss().to(device)

    for epoch in range(epochs):

        model.train()
        total_loss, total_len = 0, 0
        for x_u, x_i, y in train_dataloader:
            x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
            y_pre = model(x_u, x_i)
            loss = loss_func(y_pre, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()#根据梯度更新模型的参数。

            total_loss += loss.item() * len(y)
            total_len += len(y)
        train_loss = total_loss / total_len

        model.eval()#将模型设置为评估模式，这会禁用某些仅在训练时使用的层。
        labels, predicts = [], []
        with torch.no_grad():#使用 torch.no_grad() 上下文管理器来禁用梯度计算，这在评估时可以减少内存使用并加速计算。
            for x_u, x_i, y in test_dataloader:
                x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
                y_pre = model(x_u, x_i)
                labels.extend(y.tolist())
                predicts.extend(y_pre.tolist())
        mse = mean_squared_error(np.array(labels), np.array(predicts))
        # 存储训练损失
        train_losses.append(train_loss)

        # ...（省略测试循环代码）

        # 存储验证MSE
        val_mses.append(mse)
        print("epoch {}, train loss is {}, val mse is {}".format(epoch, train_loss, mse))
 # 可视化训练损失和验证MSE
    # 可视化训练损失和验证MSE
    plt.style.use('seaborn-darkgrid')  # 设置图表风格
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # 创建子图布局

    # 绘制训练损失
    axs[0].plot(train_losses, color='blue', linestyle='-', marker='o', label='Training Loss')
    axs[0].set_xlabel('Epoch', fontsize=12)  # 设置x轴标签
    axs[0].set_ylabel('Loss', fontsize=12)  # 设置y轴标签
    axs[0].set_title('Training Loss Over Epochs', fontsize=14)  # 设置标题
    axs[0].legend(loc='upper right', fontsize=11)  # 设置图例
    axs[0].grid(True)  # 显示网格

    # 绘制验证MSE
    axs[1].plot(val_mses, color='green', linestyle='-', marker='x', label='Validation MSE')
    axs[1].set_xlabel('Epoch', fontsize=12)
    axs[1].set_ylabel('MSE', fontsize=12)
    axs[1].set_title('Validation MSE Over Epochs', fontsize=14)
    axs[1].legend(loc='upper left', fontsize=11)
    axs[1].grid(True)

    plt.tight_layout()  # 自动调整子图参数
    plt.show()

if __name__ == '__main__':
    main()
