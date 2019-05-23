import os
import sys
import signal
import subprocess
import socket
import time

#  s = socket.socket(AF_INET, SOCK_STREAM)
#
#  host = socket.gethostname()
#  port = 9999
#
#  server

PID = -1

DEFAULT_LEADER_PORT = 12000

DEFAULT_BINARY_FOLDER = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "build"))
DEFAULT_EXPER_FOLDER = os.path.abspath(os.path.join(os.getcwd(), ".."))

ip = sys.argv[1]
port = sys.argv[2]
DEFAULT_TRAINING_PATH = os.path.abspath(sys.argv[3])
DEFAULT_TESTING_PATH = os.path.abspath(sys.argv[4])
DEFAULT_IP_PORT_FILE = os.path.abspath(sys.argv[5])



def start_leader():
    cmd = "{0} {1} {2} {3} -d {4} {5} -s {6}".format(
            os.path.join(DEFAULT_BINARY_FOLDER, "permission_net_leader"),
            ip,
            port,
            -1,
            DEFAULT_TRAINING_PATH,
            DEFAULT_TESTING_PATH,
            DEFAULT_IP_PORT_FILE)
    executing_cmd(cmd)

def start_worker(ip, port, index):
    cmd = "{0} {1} {2} {3} -d {4} -s {5}".format(
            os.path.join(DEFAULT_BINARY_FOLDER, "permission_net_worker"),
            ip,
            port, 
            index,
            DEFAULT_TRAINING_PATH,
            DEFAULT_IP_PORT_FILE)
    executing_cmd(cmd)

def executing_cmd(command: list) -> int:
    print(command)
    if type(command) == str:
        cmd = command.split(" ")
    elif type(command) == list:
        cmd = command

    pro = subprocess.Popen(cmd)
    global PID
    PID = pro.pid
    #  pro.wait()

def kill_process() -> int:
    global PID
    os.killpg(os.getpgid(PID), signal.SIGKILL)


def wait_for_leader():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((ip, int(port) + 10000))
    s.listen(5)
    while True:
        conn, addr = s.accept()
        return conn


role = "leader"
ip_port_list = []
index = -1

def init_check_role():
    global role
    global ip_port_list
    global index

    ip_port_list = []
    with open(DEFAULT_IP_PORT_FILE, "r") as fh:
        content = fh.read().split("\n")[:-1]
        for line in content:
            a1 = line.split(" ")[0]
            a2 = line.split(" ")[1]
            ip_port_list.append([a1, a2])
            print(a1, ip, a1 == ip)
            print(a2, port, a2 == port)
            if a1 == ip and a2 == port:
                role = "worker"
                index = len(ip_port_list) - 1
                print("ssss", index)

def start_workload():
    global role
    global ip_port_list
    global index

    init_check_role()

    print(ip_port_list)
    print(ip, port)
    if role == "worker":
        start_worker(ip, port, index)
    else:
        time.sleep(5)
        start_leader()

if __name__ == "__main__":
    init_check_role()

    if role != "leader":
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((ip, int(port) + 3000))
        s.listen(5)
        conn, addr = s.accept()

        start_workload()

        order = conn.recv(1)
        #  time.sleep(20)
        kill_process()
    else:
        workers_socket = []
        for i in ip_port_list:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((i[0], int(i[1]) + 3000))
            workers_socket.append(s)

        start_workload()
        time.sleep(60)
        kill_process()
        time.sleep(3)
        for s in workers_socket:
            s.send("s")
        for s in workers_socket:
            s.close()


