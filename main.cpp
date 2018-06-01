

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "net.h"

struct Object {
    cv::Rect rec;
    int class_id;
    float prob;
};

const char* class_names[] = { "background","car", "cyclist", "pedestrain"};

void PlotDetectionResult(cv::Mat& frame, ncnn::Mat& result, float show_threshold)
{
    int img_h = frame.size().height;
    int img_w = frame.size().width;

    std::vector<Object> objects;
    for (int iw = 0; iw < result.h; iw++)
    {
        Object object;
        const float *values = result.row(iw);
        object.class_id = values[0];
        object.prob = values[1];
        object.rec.x = values[2] * img_w;
        object.rec.y = values[3] * img_h;
        object.rec.width = values[4] * img_w - object.rec.x;
        object.rec.height = values[5] * img_h - object.rec.y;
        objects.push_back(object);
    }

    for (int i = 0; i < objects.size(); ++i)
    {
        Object object = objects.at(i);
        if (object.prob > show_threshold)
        {
            cv::rectangle(frame, object.rec, cv::Scalar(255, 0, 0));
            std::ostringstream pro_str;
            pro_str << object.prob;
            std::string label = std::string(class_names[object.class_id]) + ": " + pro_str.str();
            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::rectangle(frame, cv::Rect(cv::Point(object.rec.x, object.rec.y - label_size.height),
                cv::Size(label_size.width, label_size.height + baseLine)),
                cv::Scalar(255, 255, 255), CV_FILLED);
            cv::putText(frame, label, cv::Point(object.rec.x, object.rec.y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }
    }
}

ncnn::Mat Detect(const ncnn::Net& mobilenet, cv::Mat& raw_img)
{
    int img_h = raw_img.size().height;
    int img_w = raw_img.size().width;

    int input_size = 300;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(raw_img.data, ncnn::Mat::PIXEL_BGR, raw_img.cols, raw_img.rows, input_size, input_size);

    const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
    const float norm_vals[3] = { 1.0 / 127.5,1.0 / 127.5,1.0 / 127.5 };
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Mat out;

    ncnn::Extractor ex = mobilenet.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    ex.input("data", in);
    ex.extract("detection_out", out);

    return out;
}

int main(int argc, char** argv)
{
    cv::VideoCapture cap("/home/gjw/data/video/MOT01.mp4");

    if (!cap.isOpened())
    {
        std::cout << "video is not open" << std::endl;
        return -1;
    }

    cv::Mat frame;
    ncnn::Net mobilenet;

    mobilenet.load_param("/home/gjw/ncnn_mobilenet_ssd/kitti_model/mobilenet.param");
    mobilenet.load_model("/home/gjw/ncnn_mobilenet_ssd/kitti_model/mobilenet.bin");

    double sumMs = 0;
    int count = 0;

    while (1)
    {
        cap >> frame;

        double t1 = (double)cv::getTickCount();
        ncnn::Mat result = Detect(mobilenet, frame);
        double t2 = (double)cv::getTickCount();
        double t = 1000 * double(t2 - t1) / cv::getTickFrequency();
        sumMs += t;
        ++count;

        std::cout << "time = " << t << " ms, FPS = " << 1000 / t
            << ", Average time = " << sumMs / count << ", Average FPS = "<< (1000 / (sumMs / count)) << std::endl;

        PlotDetectionResult(frame, result, 0.1);
        imshow("frame", frame);

        if (cv::waitKey(1) == 'q')
            break;
    }

    return 0;
}
