#include"camera.hpp"
#include<opencv2/opencv.hpp>
#include<spdlog/spdlog.h>
Camera::Camera(std::string model):model(model)
{
    SPDLOG_INFO("camera ready!");
}

Camera::~Camera()
{
    SPDLOG_INFO("camera close!");
}
void Camera::video(){
    cv::Mat frame;
    cv::VideoCapture capture(0);
    while (1)
    {
        // capture>>frame;
        // cv::imshow("camera",frame);
        // cv::waitKey(0);
        capture >> frame; //以流形式捕获图像

        cv::namedWindow("example", 1); //创建一个窗口用于显示图像，1代表窗口适应图像的分辨率进行拉伸。
            cv::imshow("example", frame);
        int  key = cv::waitKey(30); //等待30ms
        if (key ==  int('q')) //按下q退出
        {
            break;
        }
    }
}