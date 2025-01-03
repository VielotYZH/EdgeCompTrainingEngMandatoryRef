import socket

def send_file(host, port, file_path):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 使用ipv4地址与TCP协议
    client_socket.connect((host, port))  # 连接到指定的服务器IP地址和端口

    with open(file_path, "rb") as file:
        file_data = file.read()
        client_socket.sendall(file_data)

    response = client_socket.recv(1024).decode()
    print(response)

    client_socket.close()

if __name__ == "__main__":
    file_path = "TODO" # 替换为实际场景下的文件路径
    send_file("172.17.0.2", 37373, file_path)