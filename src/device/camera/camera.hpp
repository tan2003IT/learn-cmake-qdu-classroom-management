#pragma once
#include<string>
class Camera
{
private:
    std::string model;
public:
    Camera(std::string model);
    ~Camera();
    void video();
};