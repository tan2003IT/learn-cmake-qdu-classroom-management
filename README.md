# 青岛大学教室管理系统

---
这是一个基于project_B来学cmake的一个案例，当然希望未来有一天真的能实现

## 注意！
+ 请确保您的ubuntu中有opencv-c++
+ 如果您的ubuntu中没有spdlog.h，您可以通过在终端中输入命令`sudo apt-get install libspdlog-dev`来完成
+ 在build之前，请先把文件中原有的build文件夹删除后，再build

## 思路

1.首先应从设计device开始
2.设计好了在开始设计device相应功能
3.具体应用，想想教室能干嘛呢，具有自习、上课等功能

## device--设备

+ 椅子，具有按摩、加热功能
+ 监控，录像功能
+ 多媒体，具有开关机，showPPT功能

## study -- function

教室具有上课class，自习self_learn功能

## apps

具体应用，main函数
