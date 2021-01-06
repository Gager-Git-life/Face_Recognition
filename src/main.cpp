/*
 * @Descripttion: 
 * @version: 
 * @Author: Gager
 * @Date: 2020-11-30 10:05:36
 * @LastEditors: sueRimn
 * @LastEditTime: 2020-12-08 16:15:00
 */
#include <opencv2/tracking.hpp>
#include <opencv2/opencv.hpp>
#include "face_aligner.hpp"
#include "mobileface.hpp"
#include "face_detect.h"
#include <iostream>
#include "assist.hpp"
#include <thread>

using namespace cv;
using namespace std;
using name_feature = std::pair<string, vector<float>>;


float calculate_cosine_similarity(const std::vector<float> &feature1, const std::vector<float> &feature2)
{
    assert(feature1.size() == feature2.size() && feature1.size() > 0);
    double dot = 0;
    double norm1 = 0;
    double norm2 = 0;
    int dim = feature1.size();
    for(int i = 0; i < dim; ++i)
    {
        dot += feature1[i] * feature2[i];
        norm1 += feature1[i] * feature1[i];
        norm2 += feature2[i] * feature2[i];
    }
    double similarity = dot / (sqrt(norm1 * norm2) + 1e-5);
    return float(similarity);
}


void Get_Features(TIEVD::FaceDetect &face_detect, MobilefaceNet &facenet, cv::Mat img1, vector<float> &features){

    //1.人脸检测
    auto start = chrono::steady_clock::now();
    std::vector<TIEVD::FaceInfo> face_info1 = face_detect.Detect_MaxFace(img1, 32, 3);
    auto end = chrono::steady_clock::now();
    chrono::duration<double> elapsed = end - start;
    // cout << "[INFO]>>> 最大人脸检测耗时:" << elapsed.count() << endl;

    //2.显示获取landmark
    vector<cv::Point2f> landmarks;
    for (size_t i = 0; i < 5; ++i) {
        cv::Point2f pts = cv::Point2f(face_info1[0].landmark[i*2], face_info1[0].landmark[i*2 + 1]);
        landmarks.push_back(pts);
    }
    // draw_image(img1, face_info1, landmarks);


    //3.人脸矫正
    FaceAligner facealigner;
    cv::Mat align_face;
    start = chrono::steady_clock::now();
    facealigner.align_face(img1, landmarks, align_face);
    end = chrono::steady_clock::now();
    elapsed = end - start;
    // cout << "[INFO]>>> 人脸矫正耗时:" << elapsed.count() << endl;
    // cv::imshow("align", align_face);
    // cv::waitKey(0);


    //4.获取特征值
    start = chrono::steady_clock::now();
    align_face.convertTo(align_face, CV_32FC3);
    align_face = (align_face - 127.5) / 128.0;
    facenet.GetFeature(align_face, features);
    end = chrono::steady_clock::now();
    elapsed = end - start;
    cout << "[INFO]>>> 特征提取耗时:" << elapsed.count() << endl;
    // cout << features.size() << endl;
}

vector<name_feature> Load_verify(TIEVD::FaceDetect &face_detect, MobilefaceNet &facenet){
    vector<float> feature;
    vector<name_feature> out_dict;
    string root_path = "./imgs/identify/";
    string files[] = {"oyj.jpg", "liruotong.jpg", "liuyifei.jpg", "iu.jpg", "jinliang.jpg"};
    for(int i=0; i<sizeof(files)/sizeof(files[0]); i++){
        feature.clear();
        string file_name = root_path + files[i];
        cv::Mat img = cv::imread(file_name);
        Get_Features(face_detect, facenet, img, feature);
        auto data = std::make_pair(files[i], feature);
        out_dict.push_back(data);
    }
    return out_dict;
}



int main(){

    //人脸检测。
    std::string model_path = "./models/";
    TIEVD::FaceDetect face_detect(model_path, 1, 0.7f, 0.8f, 0.9f);

    //人脸特征提取
    string madel_path = "./models/mobilefacenet.mnn";
    MobilefaceNet facenet = MobilefaceNet(madel_path, 4);

    //加载验证图像
    auto verify_data = Load_verify(face_detect, facenet);

    //
    cv::Ptr<cv::Tracker> tracker = cv::TrackerMedianFlow::create();

    VideoCapture capture(-1);
    if(!capture.isOpened()){
        cout << "[INFO]>>> 摄像头开启失败" << endl;
    }
    else{
        cout << "[INFO]>>> 摄像头开启" << endl;
    }

    cv::Mat frame;
    bool first_frame = true;
    cv::Rect2d new_box, box0, cross, display_box;
    while(1){

        capture >> frame;

        //1.人脸检测
        vector<cv::Point2f> landmarks;
        auto start = chrono::steady_clock::now();
        std::vector<TIEVD::FaceInfo> face_info1 = face_detect.Detect_MaxFace(frame, 32, 3);
        auto end = chrono::steady_clock::now();
        chrono::duration<double> elapsed = end - start;
        cout << "[INFO]>>> 最大人脸检测耗时:" << elapsed.count() << endl;

        if(face_info1.size() > 0){
            for (size_t i = 0; i < 5; ++i) {
                cv::Point2f pts = cv::Point2f(face_info1[0].landmark[i*2], face_info1[0].landmark[i*2 + 1]);
                landmarks.push_back(pts);
            }

            //2.人脸矫正
            FaceAligner facealigner;
            cv::Mat align_face;
            facealigner.align_face(frame, landmarks, align_face);

            //3.人脸特征提取
            vector<float> feature;
            align_face.convertTo(align_face, CV_32FC3);
            align_face = (align_face - 127.5) / 127.5;
            start = chrono::steady_clock::now();
            facenet.GetFeature(align_face, feature);
            end = chrono::steady_clock::now();
            elapsed = end - start;
            cout << "[INFO]>>> 特征提取耗时:" << elapsed.count() << endl;

            float max_score=0;
            string max_name;
            for(int i=0; i<verify_data.size(); i++){
                float sim = calculate_cosine_similarity(verify_data[i].second, feature);
                if(sim > max_score){
                    max_score = sim;
                    max_name = verify_data[i].first;
                }

            }

            cv::Point p1 = cv::Point(face_info1[0].bbox.xmin, face_info1[0].bbox.ymin);
            cv::Point p2 = cv::Point(face_info1[0].bbox.xmax, face_info1[0].bbox.ymax);
            // cv::rectangle(frame, p1, p2, cv::Scalar(255, 0, 0), 1);

            box0 = cv::Rect2d(p1.x, p1.y, p2.x-p1.x, p2.y-p1.y);
            if(first_frame){
                new_box = box0;
                first_frame = false;
            }
            else{
                tracker->update(frame, new_box);
            }
            cross = box0 & new_box;
            cout << cross.area() * 1.0 / box0.area() << endl;
            if(cross.area() * 1.0 / box0.area() > 0.95){
                display_box = new_box;
            }else{
                display_box = box0;
            }
            cv::rectangle(frame, display_box, cv::Scalar(0, 255, 0), 1);

            string score = to_string(max_score);
            string out_text = max_name + "--" + score;
            
            cv::putText(frame, out_text, cv::Point(face_info1[0].bbox.xmin, face_info1[0].bbox.ymin),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
        }
        
        cv::imshow("frame", frame);
        if(cv::waitKey(1) >= 0) {
            break;
        }
        
    }

}
