#include"course.hpp"
#include"self_learn.hpp"
#include<iostream>
int main(){
    int x;
    std::cout<<"上课请输入:1,其他情况自习"<<std::endl;
    std::cin>>x;
    switch (x)
    {
    case 1:
       {//上课
        Course course(30,"数学分析");
       course.init();
        break;
       }
    default:
        self_learn math_self_learn;
        math_self_learn.init();
        break;
    }
    return 0;
}