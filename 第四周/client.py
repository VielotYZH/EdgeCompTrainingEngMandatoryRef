import socket
from PIL import Image

import numpy as np


# 连接服务器
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # 使用ipv4地址与TCP协议
client_socket.connect(("192.168.6.141", 37373)) # 宿主机地址192.168.6.141, 端口号37373


# 发送图片，写死发送三次
picture_num = 0
image = Image.open(f"images/{picture_num}.jpg")
image_data = np.array(image, dtype=np.uint8).flatten()
client_socket.send(image_data.tobytes())
print(f"send picture {picture_num}")
returned_result = client_socket.recv(1) # 接收一个字节的分类结果
print(f"server return result {returned_result}")    

picture_num = 1
image = Image.open(f"images/{picture_num}.jpg")
image_data = np.array(image, dtype=np.uint8).flatten()
client_socket.send(image_data.tobytes())
print(f"send picture {picture_num}")
returned_result = client_socket.recv(1) # 接收一个字节的分类结果
print(f"server return result {returned_result}")

picture_num = 2
image = Image.open(f"images/{picture_num}.jpg")
image_data = np.array(image, dtype=np.uint8).flatten()
client_socket.send(image_data.tobytes())
print(f"send picture {picture_num}")
returned_result = client_socket.recv(1) # 接收一个字节的分类结果
print(f"server return result {returned_result}")


client_socket.close()