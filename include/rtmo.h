#ifndef RTMO_H
#define RTMO_H

#include <string>

#include "opencv2/opencv.hpp"

#include "onnxruntime_cxx_api.h"
#include "cpu_provider_factory.h"

using namespace std;
struct Landmarks
{
    float x;
    float y;
    float score;
};
struct RTMO_Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::vector<Landmarks> pts;
};

class RTMO
{
public:
    RTMO() = delete;
    RTMO(const std::string &onnx_model_path);
    virtual ~RTMO();

public:
    cv::Mat Preprocess(cv::Mat &image, int dst_width = 640, int dst_height = 640);
    void Inference(const cv::Mat &image, std::vector<RTMO_Object> &objects);

private:
    void PrintModelInfo(Ort::Session &session);

private:
    Ort::Env m_env;
    Ort::Session m_session;
    int dst_width = 416;
    int dst_height = 416;
};

#endif // !_RTM_DET_ONNX_RUNTIME_H_