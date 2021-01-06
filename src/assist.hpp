/*
 * @Descripttion: 
 * @version: 
 * @Author: Gager
 * @Date: 2020-11-30 11:17:05
 * @LastEditors: sueRimn
 * @LastEditTime: 2020-11-30 15:09:53
 */
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <iostream>
#include <vector>


void draw_image(const cv::Mat img1, const vector<TIEVD::FaceInfo> face_info1){

    cv::Mat display_img = img1.clone();

    TIEVD::FaceBox facebox = face_info1[0].bbox;
    cv::Point p1 = cv::Point(facebox.xmin, facebox.ymin);
    cv::Point p2 = cv::Point(facebox.xmax, facebox.ymax);
    cv::Scalar color = cv::Scalar(255, 0, 0);
    cv::rectangle(display_img, p1, p2, color, 1);
    cv::putText(display_img, std::to_string(facebox.score), cv::Point(facebox.xmin, facebox.ymin),
      cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255));
    for (size_t i = 0; i < 5; ++i) {
        cv::Point2f pts = cv::Point2f(face_info1[0].landmark[i*2], face_info1[0].landmark[i*2 + 1]);
        cv::circle(display_img, pts, 3, cv::Scalar(0, 255, 255), 1);
    }
    
    cv::imshow("haha", display_img);
    cv::waitKey(0);
}

