# rm-homework
RoboMaster作业，不同分支对于不同作业
## C++ 部署onnx，实现MNIST数据集的识别
> 在ROS noetic下编写detect功能包，detect实现订阅USB相机节点信息，并使用onnx推理图像并发布分类信息话题。
使用方法：
+ 首先下载usb_cam功能包：https://github.com/ros-drivers/usb_cam
+ 接着`git clone https://github.com/tan2003IT/rm-homework.git`下载detect功能包
+ 然后将上述功能包放到工作环境/src下，如图：
![image](https://github.com/tan2003IT/rm-homework/assets/116443644/be289c8a-cd2d-40c8-b9fd-b2cfcfb74098)
+ 终端进入工作环境，然后`catkin_make`,`source ./devel/setup.bash`
+ 启动`roscore`
+ 一终端输入：`roslaunch usb_cam usb_cam-test.launch`
+ 另一终端输入：`rosrun detect detect_node`
即可运行成功
