#include"sets.hpp"
#include<spdlog/spdlog.h>
#include<iostream>

Sets::Sets(int index):index(index){
    SPDLOG_INFO("入座即原神");
}
Sets::~Sets(){
    SPDLOG_INFO("离座即干饭");
}
bool Sets::accupied(){
    std::cout<<"此座未占用"<<std::endl;
    return false;
}
void Sets::heating(){
     std::cout<<"加热中"<<std::endl;
}