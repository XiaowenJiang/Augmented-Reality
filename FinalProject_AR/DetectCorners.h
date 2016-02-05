#include<iostream>
#include<opencv2\calib3d\calib3d.hpp>
#include<core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<features2d\features2d.hpp>
#include<opencv2\ml\ml.hpp>
#include<vector>
#include"Utils.h"

#ifndef __DETECTCORNERS_H__
#define __DETECTCORNERS_H__

class DetectCorners{
private:
	cv::Mat inputimg;
	cv::Size ChessBoardSize = cv::Size(9, 6);
	std::vector<cv::Point2f> allcorners;
	std::vector<cv::Point2f> fourcorners;
public:
	DetectCorners();
	~DetectCorners();
	DetectCorners(cv::Mat img);
	std::vector<cv::Point2f> getAllcorners();
	std::vector<cv::Point2f> getfourcorners();
};

#endif