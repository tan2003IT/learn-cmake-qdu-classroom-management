#include"course.hpp"
#include"spdlog/spdlog.h"
#include"camera.hpp"
#include"sets.hpp"
#include"computer.hpp"
Course::Course(int student_num,std::string class_name):student_num(student_num),class_name(class_name){
    SPDLOG_CRITICAL("this class is {}",this->class_name);
}
Course::~Course(){
    SPDLOG_CRITICAL("下课！");
}
void Course::init(){
    Camera camera("haikang");
    Sets set(12);
    Computer computer("Hisense");
    computer.showPPT();
    camera.video();
    set.heating();
}