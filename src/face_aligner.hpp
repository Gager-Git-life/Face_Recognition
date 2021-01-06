/*
 * @Descripttion: 
 * @version: 
 * @Author: Gager
 * @Date: 2020-11-27 09:16:51
 * @LastEditors: sueRimn
 * @LastEditTime: 2020-11-30 11:38:32
 */
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <assert.h>
#include <iostream>
#include <vector>

class FaceAligner{
public:
    FaceAligner();
    ~FaceAligner();
	int align_face(const cv::Mat &img_src, const std::vector<cv::Point2f> &landmark, cv::Mat &face_aligned);

private:
	cv::Mat MeanAxis0(const cv::Mat &src);
	cv::Mat ElementwiseMinus(const cv::Mat &A, const cv::Mat &B);
	cv::Mat VarAxis0(const cv::Mat &src);
	int MatrixRank(const cv::Mat &M);
	cv::Mat SimilarTransform(const cv::Mat &src, const cv::Mat &dst);
};