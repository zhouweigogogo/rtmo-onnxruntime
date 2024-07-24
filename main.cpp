#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "rtmo.h"

static int draw_unsupported(cv::Mat &rgb)
{
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}

int image_demo(RTMO &rtmo, const char *imagepath)
{
    // std::vector<cv::String> filenames;
    // cv::glob(imagepath, filenames, false);

    // for (auto img_name : filenames)
    // {
    std::vector<RTMO_Object> objects;
    cv::Mat image = cv::imread(imagepath);
    if (image.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
    rtmo.Inference(image, objects);
    // posetracker.draw(image, points);
    objects.clear();
    // cv::imwrite("../output/result.png", image);
    // }
    return 0;
}
// int webcam_demo(MoveNet &posetracker, Yolo &yolo, int cam_id)
// {
//     cv::Mat bgr;
//     cv::VideoCapture cap(cam_id, cv::CAP_V4L2);
//     std::vector<keypoint> points;
//     std::vector<Object> objects;

//     int image_id = 0;
//     while (true)
//     {
//         cap >> bgr;

//         yolo.detect_yolov8(bgr, objects);

//         for (int i = 0; i < objects.size(); i++)
//         {
//             cv::Mat tmp = bgr(objects[i].rect).clone();
//             int b_x = objects[i].rect.x;
//             int b_y = objects[i].rect.y;

//             posetracker.detect_pose(tmp, points);
//             for (int j = 0; j < points.size(); j++)
//             {
//                 points[j].x += b_x;
//                 points[j].y += b_y;
//             }
//             posetracker.draw(bgr, points);

//             points.clear();
//         }
//         yolo.draw(bgr, objects);
//         objects.clear();
//         draw_fps(bgr);
//         cv::imshow("test", bgr);
//         cv::waitKey(1);
//     }
//     return 0;
// }

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "usage: %s [mode] [path]. \n For webcam mode=0, path is cam id; \n For image demo, mode=1, path=xxx/xxx/*.jpg; \n For video, mode=2; \n For benchmark, mode=3 path=0.\n", argv[0]);
        return -1;
    }

    int mode = atoi(argv[1]);
    switch (mode)
    {
    case 0:
    {
        int cam_id = atoi(argv[2]);
        // webcam_demo(posetracker, yolo, cam_id);
        break;
    }
    case 1:
    {
        const char *images = argv[2];
        std::string rtmo_onnx_path = "../models/rtmo-t-debug-int8.onnx";
        RTMO rtmo(rtmo_onnx_path);
        image_demo(rtmo, images);
        break;
    }

    default:
    {
        fprintf(stderr, "usage: %s [mode] [path]. \n For webcam mode=0, path is cam id; \n For image demo, mode=1, path=xxx/xxx/*.jpg; \n", argv[0]);
        break;
    }
    }
}
