import gzip
import struct

import numpy as np
from sklearn.metrics import accuracy_score


def load_mnist(path, kind='train'):
    """从MNIST数据集中加载图片与标签
    
    从包含MNIST数据的目录中读取图片与标签数据, 分别以np数组的格式返回

    参数: 
        path:
            一个表示包含MNIST数据集内容的目录的字符串
        kind:
            train表示加载训练集数据, t10k表示加载测试集数据

    返回:
        两个np数组, 其内部元素为uint8类型, 分别是图片数据与标签数据
        其中图片数组是一个二维数组, 行数为图片数量, 列数为图片大小(784)
    """

    labels_path = f'{path}/{kind}-labels-idx1-ubyte.gz'
    images_path = f'{path}/{kind}-images-idx3-ubyte.gz'
    
    with gzip.open(labels_path, 'rb') as lbpath:
        # >表示使用大端字节序(即高位在前), 一个I表示一个无符号整数(uint32)4字节
        # MNIST标签集中, 第一个整数表示魔法数,表明是图片还是标签(此处为标签); 第二个整数表示标签的数量(此处为10)
        # 这两个整数不需要, 读出丢弃
        _, _ = struct.unpack('>II', lbpath.read(8)) 
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with gzip.open(images_path, 'rb') as imgpath:
        # 前两个整数处理同上, 后两个分别表示每一幅图片的行数与列数
        # 此处行列均为28, 虽为已知常数但还是从文件中读出并用变量表示更加严谨
        _, _, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), rows * cols)
    
    return images, labels

X_train, y_train = load_mnist('MNIST', kind='train')
X_test, y_test = load_mnist('MNIST', kind='t10k')

# 归一化数据
X_train = X_train / 255.0
X_test = X_test / 255.0

# 将标签转换为one-hot编码
y_train_onehot = np.zeros((y_train.size, 10))
y_train_onehot[np.arange(y_train.size), y_train] = 1
y_test_onehot = np.zeros((y_test.size, 10))
y_test_onehot[np.arange(y_test.size), y_test] = 1

class MLP:
    """
    MLP

    最简单的MLP, 一共只有三层, 包括输入层, 隐藏层与输出层

    属性:
        W1: 输入层到隐藏层的权重矩阵
        b1: 输入层到隐藏层的偏置
        W2: 隐藏层到输出层的权重矩阵
        b2: 隐藏层到输出层的偏置
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        这里采用Xviar初始化方法, 个人认为这并不好
        因为该方法属于随机初始化方式, 运行的结果无法复现
        """
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / (input_size + hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / (hidden_size + output_size))
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        """激活函数"""
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        """使用softmax函数处理多分类输出"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """前向传播"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def backward(self, X, y, learning_rate):
        """反向传播"""
        m = X.shape[0]  # 获取样本数量
        
        # 计算输出层的误差
        dz2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # 计算隐藏层的误差
        dz1 = np.dot(dz2, self.W2.T) * self.a1 * (1 - self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # 更新权重和偏置
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs, learning_rate):
        """训练模型"""
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)
            if epoch % 100 == 0:  # 每一百轮输出一次交叉熵损失
                loss = -np.sum(y * np.log(output)) / y.shape[0]
                print(f'Epoch {epoch}, Loss: {loss}')
    
    def predict(self, X):
        """预测结果"""
        return np.argmax(self.forward(X), axis=1)
    

# 定义模型参数
input_size = 784
hidden_size = 128
output_size = 10
learning_rate = 0.1
epochs = 1000

# 初始化模型
mlp = MLP(input_size, hidden_size, output_size)

# 训练模型
mlp.train(X_train, y_train_onehot, epochs, learning_rate)

# 预测
y_pred = mlp.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)  # 使用accuracy_score函数计算模型准确率
print(f'Test Accuracy: {accuracy * 100:.2f}%')