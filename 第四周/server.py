import socket
import os
import threading
import time

import numpy as np
from PIL import Image

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

    def __init__(self):
        """这里通过一个已经训练好的模型直接初始化"""
        params = np.load('mlp_params.npz')
        self.W1 = params['W1']
        self.W2 = params['W2']
        self.b1 = params['b1']
        self.b2 = params['b2']
    
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
    
    def predict(self, X):
        """预测结果"""
        return np.argmax(self.forward(X), axis=1)

    def predict_single_image(self, image):
        """预测单张图片"""
        image = image / 255.0
        prediction = self.predict(image)
        return prediction[0]


server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 使用ipv4地址与TCP协议
server_socket.bind(("172.17.0.2", 37373))
server_socket.listen(2)
print(f"Server listening on 172.17.0.2:37373")

lock = threading.Lock()

def handle_client(client_socket):
    mlp = MLP()
    for i in range(3): # 写死接收三次图片
        image_data = np.frombuffer(client_socket.recv(28 * 28), dtype=np.uint8)  # 接收图片数据
        predict_result = mlp.predict_single_image(image_data)
        client_socket.send(np.uint8(predict_result).tobytes())  # 发回预测结果
        current_image_dir = os.path.join("ec4", str(predict_result))
        with lock:
            if not os.path.exists(current_image_dir):
                os.mkdir(current_image_dir)
            image = image_data.reshape(28, 28)
            image = Image.fromarray(image, 'L')
            current_time = time.time()  # 采用图片保存时的时间为图片命名
            image.save(os.path.join(current_image_dir, str(int(round(current_time * 1000))) + '.jpg'))

while True: # 服务器永不关机!!!
    client_socket, client_addr = server_socket.accept() # 接受请求
    client_thread = threading.Thread(target=handle_client, args=(client_socket, )) # 创建线程
    client_thread.start() # 开启线程