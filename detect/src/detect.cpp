#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <cuda_provider_factory.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/String.h>

using namespace std;

int time_num = 0;
const int class_num = 10;
const int input_height = 28;
const int input_width = 28;
const int input_channel = 1;
const int batch_size = 1;

const char *input_names[] = {"input.1"}; // 这个是按照onnx模型的输入来命名的
const char *output_names[] = {"26"};     // 同理，按照输出来命名

class Classifier
{
public:
  Classifier(const char *onnx_path)
  {

    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU); // 创建一个cpu内存对象信息，描述cpu内存对象属性，OrtDeviceAllocator为cpu设备分配器，OrtMemTypeCPU：CPU 内存类型。

    /*使用之前创建的内存信息对象创建了一个 float 类型的输入张量。它将张量初始化为 input_ 数组中的数据，该数组假定包含输入图像数据。input_shape_ 数组指定输入张量的维度。*/
    this->input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, input_.data(), input_.size(), input_shape_.data(), input_shape_.size());
    this->output_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, output_.data(), output_.size(), output_shape_.data(), output_shape_.size());

    // 这行代码将 CUDA 执行提供程序添加到会话选项中。这指示 ONNX Runtime 在可用时使用 CUDA 进行推理。
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_option, 0);

    // 这行代码使用指定的 ONNX 模型路径和会话选项创建一个 ONNX 会话。会话封装了 ONNX 模型以及推理执行的各种设置。
    this->session = Ort::Session(env, onnx_path, session_option);
  }

  int set_input(cv::Mat &img)
  {
    float *input_prt = input_.data();
    for (int i = 0; i < input_height; i++)
    {
      for (int j = 0; j < input_width; j++)
      {
        float tmp = img.ptr<uchar>(i)[j];
        input_prt[i * input_width + j] = tmp; // 我们将图片像素信息传输到input_这个数组里
      }
    }
    return 0;
  }

  // 使用session方法执行推理过程

  int forward()
  {
    session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
    cout << "forward finish!" << endl;
    return 0;
  }

  int get_result(int &result)
  {
    result = std::distance(output_.begin(), std::max_element(output_.begin(), output_.end()));
    cout << "result: " << result << endl;
    return 0;
  }

private:
  std::array<float, batch_size * input_height * input_width * input_channel> input_;
  std::array<int64_t, 4> input_shape_{batch_size, input_channel, input_width, input_height}; //{1，1，28，28}
  /*定义了一个名为 input_shape_ 的 std::array 对象，类型为 int64_t。它的大小为 4，
  对应输入张量的维度：batch_size、input_channel、input_width 和 input_height。
  */
  std::array<float, batch_size * class_num> output_; // 感觉是我output_出现问题了，他只存储1帧的情况，帧数一多，就会出现段错误
  std::array<int64_t, 2> output_shape_{batch_size, class_num};

  Ort::Value input_tensor_{nullptr};  // 存储图像数据
  Ort::Value output_tensor_{nullptr}; // 存储分类结果

  Ort::SessionOptions session_option;              // 创建会话选项对象：
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"}; // 初始化 ONNX Runtime 环境：
                                                   /*
                                                   这行代码创建了一个名为 env 的 Ort::Env 对象。Ort::Env 对象是一个全局单例，用于管理 ONNX Runtime 环境，
                                                   包括内存分配和日志配置。提供的参数指定了日志级别 (ORT_LOGGING_LEVEL_WARNING) 和自定义日志标识符 ("test")。
                                                   */

  Ort::Session session{nullptr}; // session 对象用于执行推理。

  // 我想着把这两个放类里面，可是会报错，灵活变量数组放在末尾有问题
  //  const char* input_names[] = "img";
  //  const char* output_names[] = "output";
};

/*
从初学者角度来讲，session 可以理解为一个容器，它包含了 ONNX 模型以及推理执行的各种设置。

当我们使用 ONNX Runtime 执行推理时，首先需要创建一个 session。session 会将 ONNX 模型加载到内存中，并初始化推理执行的各种设置。

在上述代码中，session 对象包含了以下信息：

ONNX 模型路径
输入张量
输出张量
推理执行的各种设置
使用 session 执行推理时，我们只需要将输入数据传递给 session，session 就会自动执行推理并生成输出结果。
*/

// 换个写法：这次我们把Detect节点也设计为类
class DetectNode
{
public:
  DetectNode()
  {
    // 初始化 ROS 节点
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);

    // 订阅 usb_cam 节点发布的图像话题
    image_sub_ = it.subscribe("usb_cam/image_raw", 10, &DetectNode::imageCallback, this);

    // 发布分类结果话题
    result_pub_ = nh.advertise<std_msgs::String>("detect/result", 10);
  }

private:
  // 对opencv图像进行处理：
  // 先搞清楚usb_cam节点发的消息是一帧一帧的图片，本身就是图片，不用再用 video Capture capture(0)  >> frame了
  cv::Mat img_init(cv::Mat img)
  {
    if (img.empty())
    {
      cout << "error" << endl;
    }
    cv::Mat resize_img;
    cv::resize(img, resize_img, cv::Size(input_height, input_width));
    cv::Mat gray_img;
    cv::cvtColor(resize_img, gray_img, CV_BGR2GRAY);
    // 像素反转：
    // cv::Mat dst = 255-gray_img;
    // return dst;
    // 这个地方的像素翻转我们给改成二值化：
    cv::Mat dst;
    cv::threshold(gray_img, dst, 100, 255, CV_THRESH_BINARY_INV);
    return dst;
  }
  // 图像回调函数
  void imageCallback(const sensor_msgs::ImageConstPtr &msg)
  {

    if (msg == NULL)
    {
      return;
    }

    // 将 ROS 图像转换为 OpenCV 图像
    cv::Mat tmp = cv_bridge::toCvCopy(msg, "bgr8")->image;
    cv::Mat img = img_init(tmp);
    // 使用 ONNX 模型对图像进行分类
    int result = -1;
    cout << "图像大小：" << img.rows << "  " << img.cols << endl;
    Classifier classifier("./src/detect/src/model.onnx");
    classifier.set_input(img);
    classifier.forward();
    cout << "turn:" << time_num++ << endl; // 用来记录执行了多少回
    classifier.get_result(result);

    // 发布分类结果
    std_msgs::String classfication_msg;
    classfication_msg.data = std::to_string(result);
    result_pub_.publish(classfication_msg);
  }

  // 图像订阅器
  image_transport::Subscriber image_sub_;

  // 分类结果发布器
  ros::Publisher result_pub_;
};

int main(int argc, char **argv)
{
  // 初始化 ROS 系统
  ros::init(argc, argv, "detect");

  // 创建 DetectNode 对象
  DetectNode node;

  // 循环等待 ROS 事件
  ros::spin();

  return 0;
}