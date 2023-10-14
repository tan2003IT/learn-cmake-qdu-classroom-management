#pragma once
#include<string>
class Computer
{
private:
    std::string model;
public:
    Computer(std::string model);
    ~Computer();
    void showPPT();
};