#pragma once
#include<string>
class Course{
    private:
        int student_num;
        std::string class_name;
    public:
        Course(int student_num,std::string class_name);
        ~Course();
        void init();
};