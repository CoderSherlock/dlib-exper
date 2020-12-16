#ifndef _DLIB_DNn_SYNCER_CONFIG_H_
#define _DLIB_DNn_SYNCER_CONFIG_H_

#include <string>
#include <list>
#include <fstream>
#include "utils.h"

namespace dlib
{
class dt_config
{
public:
    std::list<std::string> training_dataset_path;
    std::list<std::string> testing_dataset_path;
    std::list<device> device_list;
    int ending_epoch;

    device_role get_role(std::string ip, int port)
    {
        for (auto i = device_list.begin(); i != device_list.end(); ++i)
        {
            if (ip == i->ip && port == i->port)
            {
                return i->role;
            }
        }
        return device_role::undecided;
    };

    device_role get_role(int number)
    {
        for (auto i = device_list.begin(); i != device_list.end(); ++i)
        {
            if (number == i->number)
            {
                return i->role;
            }
        }
        return device_role::undecided;
    };

    int get_number(std::string ip, int port)
    {
        for (auto i = device_list.begin(); i != device_list.end(); ++i)
        {
            if (ip == i->ip && port == i->port)
            {
                return i->number;
            }
        }
        return -1;
    };

    device get_my_master(device me)
    {
        std::string ip = me.ip;
        int port = me.port;
        int my_leader_number = -1;
        for (auto i = device_list.begin(); i != device_list.end(); ++i)
        {
            if (ip == i->ip && port == i->port)
            {
                my_leader_number = i->master;
                break;
            }
        }
        for (auto i = device_list.begin(); i != device_list.end(); ++i)
        {
            if (i->number == my_leader_number)
            {
                return *i;
            }
        }
        return device();
    };

    int read_config(char *config_path)
    {
        std::ifstream f;
        f.open(config_path);
        int training_lines, testing_lines, device_lines;

        // Read header (line number of different records)
        f >> training_lines >> testing_lines >> device_lines >> this->ending_epoch;

        std::string tmp;
        while (training_lines--)
        {
            f >> tmp;
            this->training_dataset_path.push_back(tmp);
        }

        while (testing_lines--)
        {
            f >> tmp;
            this->testing_dataset_path.push_back(tmp);
        }

        int number = 0, port, role, master;
        std::string ip;
        while (device_lines--)
        {
            f >> number >> ip >> port >> role >> master;
            device temp(number, ip, port, device_role(role), master, 0, device_sync_type(1));
            device_list.push_back(temp);
        }

        f.close();
        return 1;
    }
};
} // namespace dlib

#endif