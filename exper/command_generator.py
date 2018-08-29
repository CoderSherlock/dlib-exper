import sys

ll = [
["192.168.0.103", "13001"],
["192.168.0.104", "13001"],
["192.168.0.105", "13001"],
["192.168.0.106", "13001"]
]

local = [
["127.0.0.1", "13001"],
["127.0.0.1", "13002"],
["127.0.0.1", "13003"],
["127.0.0.1", "13004"],
]

def from_list(ip_port_list):
    ip_port_list = sorted(ip_port_list)
    for i in range(1, len(ip_port_list)):
        print(get_slave_command(len(ip_port_list) - 1, i, ip_port_list[i][0], ip_port_list[i][1], ip_port_list[0][0], ip_port_list[0][1]))

    print(get_master_command(len(ip_port_list) - 1, ip_port_list[0][0], ip_port_list[0][1], [x[0] for x in ip_port_list], [x[1] for x in ip_port_list]))



def get_master_command(slavecount, masterip, masterport, ips, ports):
    ret = "./dnn_master master " + str(masterip) + " " + str(masterport) + " 0 -d /home/dlib/data/mnist -s " + str(slavecount) + " "
    for i in range(0, len(ips)):
        ret += str(ips[i]) + " " + str(ports[i]) + " "
    return ret

def get_slave_command(slavecount, index, ip, port, masterip, masterport):
    return "./dnn_master slave " + str(ip) + " " + str(port) + " " + str(index) + " -d /home/dlib/data/mnist -s " + str(slavecount)


from_list(local)
