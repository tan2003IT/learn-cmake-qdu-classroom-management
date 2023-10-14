#include"computer.hpp"
#include<spdlog/spdlog.h>
#include<iostream>
Computer::Computer(std::string model):model(model)
{
    SPDLOG_INFO("computer turn on");
}

Computer::~Computer()
{
    SPDLOG_INFO("computer turn off");
}
void Computer::showPPT(){
    std::cout<<"this is a slide"<<std::endl;
}