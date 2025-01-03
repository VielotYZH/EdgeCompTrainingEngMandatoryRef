import socket

def start_server(host, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 使用ipv4地址与TCP协议
    server_socket.bind((host, port))  # 绑定ip地址与端口
    server_socket.listen(2)  # 开始监听, 最多两个连接
    print(f"Server listening on {host}:{port}")

    clients = {}  # 记录请求的字典, 客户端ip地址为键, socket对象为值

    while len(clients) < 2:
        client_socket, client_address = server_socket.accept()
        clients[client_address] = client_socket
        print(f"Connected to {client_address}")

    for client_address, client_socket in clients.items():
        file_data = client_socket.recv(26624)  # 这里每次读取的字节数尽量写大一点, 保证每次都能接收完整图片
        file_name = f"{client_address}_image.png"
          
        with open(file_name, "wb") as file:
            file.write(file_data)
      
        print(f"Received image from {client_address}: {file_name}")
        client_socket.send("OK".encode())

    server_socket.close()

if __name__ == "__main__":
    start_server("172.17.0.2", 37373)