#include "rtmo.h"

#include <iostream>
#include <thread>

const int num_joints = 17;

RTMO::RTMO(const std::string &onnx_model_path)
    : m_session(nullptr),
      m_env(nullptr)
{
    // std::wstring modelPath = std::wstring(onnx_model_path.begin(), onnx_model_path.end());

    m_env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "rtmo-onnxrumtime-cpu");

    int cpu_processor_num = std::thread::hardware_concurrency();
    cpu_processor_num /= 2;

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(cpu_processor_num);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetLogSeverityLevel(4);

    std::cout << "onnxruntime inference try to use CPU Device" << std::endl;
    OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 0);

    m_session = Ort::Session(m_env, onnx_model_path.c_str(), session_options);
}

RTMO::~RTMO()
{
}

bool cmp(RTMO_Object b1, RTMO_Object b2)
{
    return b1.prob > b2.prob;
}
void nms(std::vector<RTMO_Object> &input_boxes, float NMS_THRESH)
{
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = input_boxes.at(i).rect.area();
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = std::max(input_boxes.at(i).rect.x, input_boxes.at(j).rect.x);
            float yy1 = std::max(input_boxes.at(i).rect.y, input_boxes.at(j).rect.y);
            float xx2 = std::min(input_boxes.at(i).rect.x + input_boxes.at(i).rect.width, input_boxes.at(j).rect.x + input_boxes.at(j).rect.width);
            float yy2 = std::min(input_boxes.at(i).rect.y + input_boxes.at(i).rect.height, input_boxes.at(j).rect.y + input_boxes.at(j).rect.height);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}

void RTMO::Inference(const cv::Mat &image, std::vector<RTMO_Object> &objects)
{

    // get input and output info
    int input_nodes_num = m_session.GetInputCount();
    int output_nodes_num = m_session.GetOutputCount();
    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;
    Ort::AllocatorWithDefaultOptions allocator;
    int input_h = 0;
    int input_w = 0;

    // query input data format
    for (int i = 0; i < input_nodes_num; i++)
    {
        auto input_name = m_session.GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());
        auto inputShapeInfo = m_session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        int ch = inputShapeInfo[1];
        input_h = inputShapeInfo[2];
        input_w = inputShapeInfo[3];
        std::cout << "input format: " << ch << "x" << input_h << "x" << input_w << std::endl;
    }

    int out_h = 0; // 845
    int out_w = 0; // 56
    for (int i = 0; i < output_nodes_num; i++)
    {
        auto output_name = m_session.GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(output_name.get());
        auto outShapeInfo = m_session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        for (int j = 0; j < outShapeInfo.size(); j++)
            std::cout << outShapeInfo[j] << std::endl;
        out_h = outShapeInfo[1];
        out_w = outShapeInfo[2];
        std::cout << "output format: " << out_h << "x" << out_w << std::endl;
    }

    int64 pre_start = cv::getTickCount();

    // preprocess
    float scale = std::min((float)(dst_width) / image.cols, (float)(dst_height) / image.rows);
    float ox = (dst_width - scale * image.cols) / 2.0;
    float oy = (dst_height - scale * image.rows) / 2.0;
    cv::Mat M = (cv::Mat_<float>(2, 3) << scale, 0., ox, 0., scale, oy);

    cv::Mat img_pre;
    cv::warpAffine(image, img_pre, M, cv::Size(dst_width, dst_height), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::Mat IM;
    cv::invertAffineTransform(M, IM);

    cv::cvtColor(img_pre, img_pre, cv::COLOR_BGR2RGB);

    cv::Mat process_img;
    img_pre.convertTo(process_img, CV_32F);

    // std::cout << process_img.at<cv::Vec3f>(251, 215) << std::endl;
    cv::Mat blob = cv::dnn::blobFromImage(process_img, 1.0, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true, false);
    size_t tpixels = input_h * input_w * 3;
    std::array<int64_t, 4> input_shape_info{1, 3, input_h, input_w};

    // set input data and inference
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
    const std::array<const char *, 1> inputNames = {input_node_names[0].c_str()};
    const std::array<const char *, 1> outNames = {output_node_names[0].c_str()};

    int64 pre_end = cv::getTickCount();
    double pre_time = (pre_end - pre_start) / cv::getTickFrequency();

    std::cout << "pre time: " << pre_time << " s" << std::endl;

    int64 infer_start = cv::getTickCount();
    std::vector<Ort::Value> ort_outputs;
    try
    {
        ort_outputs = m_session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
    }
    catch (std::exception e)
    {
        std::cout << e.what() << std::endl;
    }
    int64 infer_end = cv::getTickCount();
    double infer_time = (infer_end - infer_start) / cv::getTickFrequency();

    std::cout << "infer time: " << infer_time << " s" << std::endl;

    // 845*(51+5)
    const float *pdata = ort_outputs[0].GetTensorMutableData<float>();

    cv::Mat dout(845, 56, CV_32F, (float *)pdata);

    int64 post_start = cv::getTickCount();

    cv::Mat output = dout.t(); // 56*2000
    cv::Mat scores = output.row(4);
    for (int i = 0; i < scores.cols; i++)
    {
        float score = scores.at<float>(0, i);

        if (score < 0.25)
        {

            continue;
        }
        else
        {
            RTMO_Object obj;
            cv::Mat tmp = output.col(i);
            float left = tmp.at<float>(0, 0) * IM.at<float>(0, 0) + IM.at<float>(0, 2);
            float top = tmp.at<float>(1, 0) * IM.at<float>(1, 1) + IM.at<float>(1, 2);
            float right = tmp.at<float>(2, 0) * IM.at<float>(0, 0) + IM.at<float>(0, 2);
            float bottom = tmp.at<float>(3, 0) * IM.at<float>(1, 1) + IM.at<float>(1, 2);
            obj.rect.x = left;
            obj.rect.y = top;
            obj.rect.width = right - left;
            obj.rect.height = bottom - top;
            obj.prob = score;

            obj.pts.resize(num_joints);
            for (int j = 0; j < num_joints; j++)
            {
                float point_x = tmp.at<float>(4 + j * 3 + 1, 0) * IM.at<float>(0, 0) + IM.at<float>(0, 2);
                float point_y = tmp.at<float>(4 + j * 3 + 2, 0) * IM.at<float>(1, 1) + IM.at<float>(1, 2);
                float point_score = tmp.at<float>(4 + j * 3 + 3, 0) * IM.at<float>(0, 0) + IM.at<float>(0, 2);
                // std::cout << point_x << " " << point_y << " " << point_score << std::endl;
                obj.pts[j].x = point_x;
                obj.pts[j].y = point_y;
                obj.pts[j].score = point_score;
            }
            objects.push_back(obj);
        }
    }
    std::sort(objects.begin(), objects.end(), cmp);
    nms(objects, 0.35f);

    int64 post_end = cv::getTickCount();
    double post_time = (post_end - post_start) / cv::getTickFrequency();

    std::cout << "post time: " << post_time << " s" << std::endl;
    // draw
    int color_index[][3] = {
        {255, 0, 0},
        {0, 0, 255},
        {255, 0, 0},
        {0, 0, 255},
        {255, 0, 0},
        {0, 0, 255},
        {0, 255, 0},
        {255, 0, 0},
        {255, 0, 0},
        {0, 0, 255},
        {0, 0, 255},
        {0, 255, 0},
        {255, 0, 0},
        {255, 0, 0},
        {255, 0, 0},
        {0, 0, 255},
        {0, 0, 255},
        {0, 0, 255},
    };

    for (int i = 0; i < objects.size(); i++)
    {
        auto obj = objects[i];
        std::cout << obj.rect << std::endl;
        cv::rectangle(image, obj.rect, cv::Scalar(0, 0, 255), 2, 8, 0);
        for (int j = 0; j < num_joints; j++)
        {
            cv::circle(image, cv::Point(obj.pts[j].x, obj.pts[j].y), 3, cv::Scalar(100, 255, 150), -1);
        }
    }
    cv::imwrite("output.jpg", image);
}