/*
 * @Descripttion: 
 * @version: 
 * @Author: Gager
 * @Date: 2020-11-26 15:51:24
 * @LastEditors: sueRimn
 * @LastEditTime: 2020-11-30 11:40:24
 */

#ifndef mobileface_hpp
#define mobileface_hpp

#pragma once

#include <opencv2/opencv.hpp>
#include "ImageProcess.hpp"
#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "Tensor.hpp"
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>


using namespace std;



class MobilefaceNet {
public:
    MobilefaceNet(const std::string &mnn_path, int num_thread_ = 4);

    ~MobilefaceNet();
    void GetFeature(const cv::Mat &frame, vector<float> &feature);
    cv::Mat Get_Resize_Croped_Img(cv::Mat frame, cv::Point pt1, cv::Point pt2, cv::Point &s_point, cv::Size &croped_wh);

private:

    std::shared_ptr<MNN::Interpreter> mobileface_interpreter;
    MNN::Session *mobileface_session = nullptr;
    MNN::Tensor *input_tensor = nullptr;
    MNN::Tensor *nchw_Tensor = nullptr;
    const int INPUT_SIZE = 112;

};

#endif /* mobileface_hpp */
